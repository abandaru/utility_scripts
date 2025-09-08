#!/usr/bin/env python3
"""
COBOL Caller-Callee Scanner for Manta Data Lineage - Version 2
==============================================================

This script scans COBOL files to identify caller-callee relationships between
main programs and sub-programs, generating output compatible with IBM Manta
Data Lineage scanner.

Key Features:
- Recursive directory scanning
- COBOL program identification  
- CALL statement parsing (static and dynamic)
- Enhanced variable-based call tracking with VALUE clause support
- Manta-compatible CSV output (PROGRAM;VARIABLE;SUB-PROGRAM-NAME format)
- Comprehensive error handling and logging
- Multiple output formats for validation and debugging

Version: 2.0.0
Author: Adi Bandaru
Manta Compatibility: IBM Manta Data Lineage Scanner Guide compliant
"""

import re
import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cobol_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CobolProgram:
    """Represents a COBOL program with its metadata"""
    name: str
    file_path: str
    program_id: Optional[str] = None
    division_type: str = "main"
    variables: List[str] = None
    copybooks: List[str] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.copybooks is None:
            self.copybooks = []

@dataclass
class CallRelationship:
    """Represents a caller-callee relationship"""
    caller_program: str
    caller_file: str
    callee_program: Optional[str]
    call_type: str  # 'static', 'dynamic', 'copybook'
    call_variable: Optional[str] = None
    line_number: int = 0
    call_statement: str = ""

@dataclass
class ScanResults:
    """Container for scan results"""
    programs: List[CobolProgram]
    relationships: List[CallRelationship]
    errors: List[str]
    statistics: Dict[str, int]

class CobolCallScanner:
    """
    Main scanner class for identifying COBOL caller-callee relationships
    Compatible with IBM Manta Data Lineage Scanner requirements
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # COBOL file extensions
        self.cobol_extensions = {'.cbl', '.cob', '.cobol', '.cpy', '.copy'}
        
        # Enhanced regex patterns for COBOL parsing
        self.patterns = {
            'program_id': re.compile(
                r'^\s*PROGRAM-ID\.\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            ),
            'call_static': re.compile(
                r'^\s*CALL\s+["\']([A-Z0-9\-_]+)["\']',
                re.IGNORECASE | re.MULTILINE
            ),
            'call_dynamic': re.compile(
                r'^\s*CALL\s+([A-Z0-9\-_]+)(?:\s+USING|\s*$|\s*\.)',
                re.IGNORECASE | re.MULTILINE
            ),
            'copy_statement': re.compile(
                r'^\s*COPY\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            ),
            'working_storage': re.compile(
                r'^\s*WORKING-STORAGE\s+SECTION',
                re.IGNORECASE | re.MULTILINE
            ),
            'data_division': re.compile(
                r'^\s*DATA\s+DIVISION',
                re.IGNORECASE | re.MULTILINE
            ),
            'procedure_division': re.compile(
                r'^\s*PROCEDURE\s+DIVISION',
                re.IGNORECASE | re.MULTILINE
            ),
            'variable_definition': re.compile(
                r'^\s*\d{2}\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            ),
            'move_statement': re.compile(
                r'^\s*MOVE\s+["\']([A-Z0-9\-_]+)["\']\s+TO\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            ),
            # Additional patterns for better COBOL parsing
            'move_literal': re.compile(
                r'^\s*MOVE\s+([A-Z0-9\-_]+)\s+TO\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            ),
            'value_clause': re.compile(
                r'^\s*\d{2}\s+([A-Z0-9\-_]+).*?VALUE\s+["\']([A-Z0-9\-_]+)["\']',
                re.IGNORECASE | re.MULTILINE
            ),
            # Pattern for VALUE without quotes
            'value_clause_unquoted': re.compile(
                r'^\s*\d{2}\s+([A-Z0-9\-_]+).*?VALUE\s+([A-Z0-9\-_]+)',
                re.IGNORECASE | re.MULTILINE
            )
        }
        
        # Statistics tracking
        self.stats = {
            'files_scanned': 0,
            'programs_found': 0,
            'static_calls': 0,
            'dynamic_calls': 0,
            'copy_statements': 0,
            'errors': 0
        }
        
        # Variable to program name mappings for dynamic call resolution
        self.variable_mappings: Dict[str, str] = {}

    def scan_directory(self, directory: str, recursive: bool = True) -> ScanResults:
        """
        Scan directory for COBOL files and analyze caller-callee relationships
        
        Args:
            directory: Path to directory containing COBOL files
            recursive: Whether to scan subdirectories
            
        Returns:
            ScanResults object containing all findings
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        logger.info(f"Starting scan of directory: {directory}")
        logger.info(f"Recursive scan: {recursive}")
        
        programs: List[CobolProgram] = []
        relationships: List[CallRelationship] = []
        errors: List[str] = []
        
        # Find all COBOL files
        cobol_files = self._find_cobol_files(directory_path, recursive)
        logger.info(f"Found {len(cobol_files)} COBOL files")
        
        # First pass: Parse all programs and build variable mappings
        for file_path in cobol_files:
            try:
                program = self._parse_cobol_file(file_path)
                if program:
                    programs.append(program)
                    self.stats['programs_found'] += 1
                    
                self.stats['files_scanned'] += 1
                
            except Exception as e:
                error_msg = f"Error parsing {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                self.stats['errors'] += 1
        
        # Second pass: Analyze call relationships
        for file_path in cobol_files:
            try:
                file_relationships = self._analyze_call_relationships(file_path)
                relationships.extend(file_relationships)
                
            except Exception as e:
                error_msg = f"Error analyzing calls in {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        results = ScanResults(
            programs=programs,
            relationships=relationships,
            errors=errors,
            statistics=self.stats.copy()
        )
        
        logger.info(f"Scan completed. Found {len(programs)} programs, {len(relationships)} relationships")
        return results

    def _find_cobol_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all COBOL files in directory"""
        cobol_files = []
        
        if recursive:
            for ext in self.cobol_extensions:
                cobol_files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in self.cobol_extensions:
                cobol_files.extend(directory.glob(f"*{ext}"))
        
        return sorted(cobol_files)

    def _parse_cobol_file(self, file_path: Path) -> Optional[CobolProgram]:
        """Parse a COBOL file to extract program information and build variable mappings"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None
        
        # Extract program ID
        program_id_match = self.patterns['program_id'].search(content)
        program_id = program_id_match.group(1) if program_id_match else file_path.stem
        
        # Extract variables from working storage
        variables = []
        if self.patterns['working_storage'].search(content):
            var_matches = self.patterns['variable_definition'].findall(content)
            variables.extend(var_matches)
        
        # Extract copybooks
        copybooks = []
        copy_matches = self.patterns['copy_statement'].findall(content)
        copybooks.extend(copy_matches)
        
        # Build variable to program mappings from MOVE statements (quoted literals)
        move_matches = self.patterns['move_statement'].findall(content)
        for program_name, variable_name in move_matches:
            full_key = f"{program_id}.{variable_name}"
            self.variable_mappings[full_key] = program_name
            # Also add without program prefix for cross-program resolution
            self.variable_mappings[variable_name] = program_name
        
        # Handle MOVE statements with unquoted literals
        # MOVE PROGRAM-NAME TO VARIABLE
        literal_moves = self.patterns['move_literal'].findall(content)
        for source_value, target_variable in literal_moves:
            # Only consider if source looks like a program name (all caps, no quotes)
            if (source_value.isupper() and 
                not source_value.startswith('"') and 
                len(source_value) <= 8 and  # COBOL program names typically <= 8 chars
                not source_value.isdigit()):  # Not a numeric literal
                full_key = f"{program_id}.{target_variable}"
                self.variable_mappings[full_key] = source_value
                self.variable_mappings[target_variable] = source_value
        
        # Handle VALUE clauses in variable definitions (quoted)
        # Example: 01 WS-PROG-NAME PIC X(8) VALUE 'SUBPROG'.
        value_matches = self.patterns['value_clause'].findall(content)
        for variable_name, program_name in value_matches:
            full_key = f"{program_id}.{variable_name}"
            self.variable_mappings[full_key] = program_name
            self.variable_mappings[variable_name] = program_name
            
        # Handle VALUE clauses without quotes
        # Example: 01 WS-PROG-NAME PIC X(8) VALUE SUBPROG.
        value_unquoted_matches = self.patterns['value_clause_unquoted'].findall(content)
        for variable_name, program_name in value_unquoted_matches:
            # Skip if already captured by quoted version or if it's a numeric/special value
            if (f"{program_id}.{variable_name}" not in self.variable_mappings and
                program_name.isupper() and 
                not program_name.isdigit() and
                program_name not in ['ZERO', 'ZEROS', 'SPACE', 'SPACES', 'LOW-VALUE', 'HIGH-VALUE']):
                full_key = f"{program_id}.{variable_name}"
                self.variable_mappings[full_key] = program_name
                self.variable_mappings[variable_name] = program_name
        
        return CobolProgram(
            name=program_id,
            file_path=str(file_path),
            program_id=program_id,
            variables=variables,
            copybooks=copybooks
        )

    def _analyze_call_relationships(self, file_path: Path) -> List[CallRelationship]:
        """Analyze CALL statements in a COBOL file"""
        relationships = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return relationships
        
        # Get program name
        content = ''.join(lines)
        program_id_match = self.patterns['program_id'].search(content)
        caller_program = program_id_match.group(1) if program_id_match else file_path.stem
        
        # Analyze each line
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('*') or line.startswith('//'):
                continue
            
            # Check for static CALL statements first
            static_match = self.patterns['call_static'].search(line)
            if static_match:
                callee_program = static_match.group(1)
                relationships.append(CallRelationship(
                    caller_program=caller_program,
                    caller_file=str(file_path),
                    callee_program=callee_program,
                    call_type='static',
                    line_number=line_num,
                    call_statement=line
                ))
                self.stats['static_calls'] += 1
                continue
            
            # Check for dynamic CALL statements (only if not static)
            dynamic_match = self.patterns['call_dynamic'].search(line)
            if dynamic_match:
                variable_name = dynamic_match.group(1)
                
                # Try to resolve variable to program name
                full_key = f"{caller_program}.{variable_name}"
                callee_program = self.variable_mappings.get(full_key) or self.variable_mappings.get(variable_name)
                
                relationships.append(CallRelationship(
                    caller_program=caller_program,
                    caller_file=str(file_path),
                    callee_program=callee_program,
                    call_type='dynamic',
                    call_variable=variable_name,
                    line_number=line_num,
                    call_statement=line
                ))
                self.stats['dynamic_calls'] += 1
                continue
            
            # Check for COPY statements
            copy_match = self.patterns['copy_statement'].search(line)
            if copy_match:
                copybook_name = copy_match.group(1)
                relationships.append(CallRelationship(
                    caller_program=caller_program,
                    caller_file=str(file_path),
                    callee_program=copybook_name,
                    call_type='copybook',
                    line_number=line_num,
                    call_statement=line
                ))
                self.stats['copy_statements'] += 1
        
        return relationships

    def export_manta_format(self, results: ScanResults) -> str:
        """
        Export results in Manta Data Lineage compatible format
        
        Generates CSV file in format: PROGRAM;VARIABLE;SUB-PROGRAM-NAME
        Specifically for CALL statements where sub-program is defined by a variable
        According to IBM Manta Data Lineage Scanner Guide
        
        Returns path to generated file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate call targets file for Manta
        call_targets_file = self.output_dir / f"cobol_call_targets_{timestamp}.csv"
        
        with open(call_targets_file, 'w', newline='', encoding='utf-8') as f:
            # Header as required by Manta (quoted as per documentation)
            f.write('"Calling program name";"Calling variable name";"Target program name"\n')
            
            # Write dynamic call relationships only (as per Manta requirements)
            # Format: PROGRAM;VARIABLE;SUB-PROGRAM-NAME (no quotes on data)
            dynamic_calls_written = 0
            for rel in results.relationships:
                if rel.call_type == 'dynamic' and rel.callee_program and rel.call_variable:
                    # Write without quotes on the data (only header is quoted)
                    f.write(f"{rel.caller_program};{rel.call_variable};{rel.callee_program}\n")
                    dynamic_calls_written += 1
            
            # If no dynamic calls found, add a comment line for clarity
            if dynamic_calls_written == 0:
                f.write('# No dynamic CALL statements with resolved variables found\n')
        
        logger.info(f"Manta call targets file generated: {call_targets_file}")
        logger.info(f"Dynamic calls exported for Manta: {dynamic_calls_written}")
        return str(call_targets_file)

    def export_detailed_report(self, results: ScanResults) -> str:
        """Export detailed analysis report in JSON format with Manta compatibility information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"cobol_analysis_report_{timestamp}.json"
        
        # Categorize relationships for better reporting
        static_calls = [r for r in results.relationships if r.call_type == 'static']
        dynamic_calls = [r for r in results.relationships if r.call_type == 'dynamic']
        copybook_calls = [r for r in results.relationships if r.call_type == 'copybook']
        
        # Identify unresolved dynamic calls (important for Manta)
        unresolved_dynamic = [r for r in dynamic_calls if not r.callee_program]
        resolved_dynamic = [r for r in dynamic_calls if r.callee_program]
        
        report_data = {
            'scan_metadata': {
                'timestamp': timestamp,
                'scanner_version': '2.0.0',
                'manta_compatible': True,
                'statistics': results.statistics,
                'manta_compatibility': {
                    'dynamic_calls_total': len(dynamic_calls),
                    'dynamic_calls_resolved': len(resolved_dynamic),
                    'dynamic_calls_unresolved': len(unresolved_dynamic),
                    'dynamic_calls_for_manta': len([r for r in resolved_dynamic if r.call_variable]),
                    'static_calls_total': len(static_calls),
                    'copybook_calls_total': len(copybook_calls),
                    'variable_mappings_found': len(self.variable_mappings)
                }
            },
            'programs': [asdict(prog) for prog in results.programs],
            'relationships': {
                'static_calls': [asdict(rel) for rel in static_calls],
                'dynamic_calls_resolved': [asdict(rel) for rel in resolved_dynamic],
                'dynamic_calls_unresolved': [asdict(rel) for rel in unresolved_dynamic],
                'copybook_calls': [asdict(rel) for rel in copybook_calls]
            },
            'unresolved_dynamic_calls': [
                {
                    'caller_program': rel.caller_program,
                    'variable_name': rel.call_variable,
                    'line_number': rel.line_number,
                    'call_statement': rel.call_statement,
                    'file_path': rel.caller_file,
                    'manta_entry_suggestion': f"{rel.caller_program};{rel.call_variable};TARGET-PROGRAM-NAME"
                }
                for rel in unresolved_dynamic
            ],
            'variable_mappings': self.variable_mappings,
            'errors': results.errors
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed report generated: {report_file}")
        if unresolved_dynamic:
            logger.warning(f"Found {len(unresolved_dynamic)} unresolved dynamic calls - check detailed report")
        return str(report_file)

    def export_program_summary(self, results: ScanResults) -> str:
        """Export summary of programs by directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"program_summary_{timestamp}.csv"
        
        # Group programs by directory
        directory_summary = {}
        for program in results.programs:
            dir_path = Path(program.file_path).parent
            if str(dir_path) not in directory_summary:
                directory_summary[str(dir_path)] = []
            directory_summary[str(dir_path)].append(program)
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Directory',
                'Program Name',
                'Program ID',
                'File Path',
                'Variables Count',
                'Copybooks Count'
            ])
            
            for directory, programs in sorted(directory_summary.items()):
                for program in programs:
                    writer.writerow([
                        directory,
                        program.name,
                        program.program_id,
                        program.file_path,
                        len(program.variables),
                        len(program.copybooks)
                    ])
        
        logger.info(f"Program summary generated: {summary_file}")
        return str(summary_file)

    def export_all_relationships(self, results: ScanResults) -> str:
        """Export all caller-callee relationships (not just dynamic ones for Manta)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_rels_file = self.output_dir / f"all_relationships_{timestamp}.csv"
        
        with open(all_rels_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Header
            writer.writerow([
                'Caller Program',
                'Callee Program', 
                'Call Type',
                'Variable Name',
                'Line Number',
                'Call Statement'
            ])
            
            # Write all relationships
            for rel in results.relationships:
                writer.writerow([
                    rel.caller_program,
                    rel.callee_program or 'UNRESOLVED',
                    rel.call_type,
                    rel.call_variable or '',
                    rel.line_number,
                    rel.call_statement.strip()[:100]  # Truncate long statements
                ])
        
        logger.info(f"All relationships exported: {all_rels_file}")
        return str(all_rels_file)

    def export_manta_config_template(self, results: ScanResults) -> str:
        """Generate Manta configuration template with unresolved dynamic calls"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.output_dir / f"manta_unresolved_calls_{timestamp}.txt"
        
        # Find unresolved dynamic calls
        unresolved = [r for r in results.relationships 
                     if r.call_type == 'dynamic' and not r.callee_program]
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("# Manta Data Lineage - Unresolved Dynamic CALL Configuration\n")
            f.write("# Add these entries to your cobol.call.targets.file\n")
            f.write("# Format: CALLING-PROGRAM;VARIABLE-NAME;TARGET-PROGRAM-NAME\n")
            f.write("# Reference: IBM Manta Data Lineage COBOL Scanner Guide\n\n")
            
            if unresolved:
                f.write('"Calling program name";"Calling variable name";"Target program name"\n')
                for rel in unresolved:
                    f.write(f"# TODO: Resolve this dynamic call\n")
                    f.write(f"{rel.caller_program};{rel.call_variable};TARGET-PROGRAM-NAME\n")
                    f.write(f"# Found in: {rel.caller_file}:{rel.line_number}\n")
                    f.write(f"# Statement: {rel.call_statement.strip()}\n\n")
            else:
                f.write("# ✅ No unresolved dynamic calls found!\n")
                f.write("# All dynamic calls have been resolved automatically.\n")
                f.write("# Your COBOL code is ready for Manta Data Lineage analysis.\n")
        
        logger.info(f"Manta configuration template generated: {config_file}")
        return str(config_file)

def main():
    """Main entry point for the scanner"""
    parser = argparse.ArgumentParser(
        description='COBOL Caller-Callee Scanner for Manta Data Lineage v2.0',
        epilog='Generates IBM Manta Data Lineage compatible output files'
    )
    parser.add_argument(
        'directory',
        help='Directory containing COBOL files to scan'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively scan subdirectories (default: True)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Output directory for generated files (default: output)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize scanner
        scanner = CobolCallScanner(args.output_dir)
        
        # Perform scan
        results = scanner.scan_directory(args.directory, args.recursive)
        
        # Generate all output files
        manta_file = scanner.export_manta_format(results)
        report_file = scanner.export_detailed_report(results)
        summary_file = scanner.export_program_summary(results)
        all_rels_file = scanner.export_all_relationships(results)
        config_template = scanner.export_manta_config_template(results)
        
        # Calculate specific statistics for Manta compatibility
        dynamic_calls = [r for r in results.relationships if r.call_type == 'dynamic']
        resolved_dynamic = [r for r in dynamic_calls if r.callee_program]
        unresolved_dynamic = [r for r in dynamic_calls if not r.callee_program]
        manta_ready_calls = [r for r in resolved_dynamic if r.call_variable]
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("COBOL SCANNER v2.0 - MANTA DATA LINEAGE COMPATIBLE")
        print("="*70)
        print(f"Files scanned: {results.statistics['files_scanned']}")
        print(f"Programs found: {results.statistics['programs_found']}")
        print(f"Static calls: {results.statistics['static_calls']}")
        print(f"Dynamic calls: {results.statistics['dynamic_calls']}")
        print(f"  - Resolved: {len(resolved_dynamic)}")
        print(f"  - Unresolved: {len(unresolved_dynamic)}")
        print(f"  - Ready for Manta: {len(manta_ready_calls)}")
        print(f"Copy statements: {results.statistics['copy_statements']}")
        print(f"Variable mappings built: {len(scanner.variable_mappings)}")
        print(f"Errors: {results.statistics['errors']}")
        
        print("\n📊 Manta Data Lineage Compatibility:")
        if manta_ready_calls:
            print(f"✅ Dynamic calls exported to Manta: {len(manta_ready_calls)}")
        else:
            print("⚠️  No dynamic calls found for Manta export")
            
        if unresolved_dynamic:
            print(f"⚠️  Unresolved dynamic calls: {len(unresolved_dynamic)}")
            print("   → Check manta_unresolved_calls_*.txt for manual resolution")
        else:
            print("✅ All dynamic calls resolved automatically!")
        
        print("\n📁 Generated files:")
        print(f"├─ Manta call targets: {Path(manta_file).name}")
        print(f"├─ All relationships: {Path(all_rels_file).name}")
        print(f"├─ Detailed report: {Path(report_file).name}")
        print(f"├─ Program summary: {Path(summary_file).name}")
        print(f"└─ Manta config template: {Path(config_template).name}")
        
        if results.errors:
            print(f"\n⚠️  Errors encountered: {len(results.errors)}")
            for error in results.errors[:3]:  # Show first 3 errors
                print(f"   • {error}")
            if len(results.errors) > 3:
                print(f"   • ... and {len(results.errors) - 3} more errors")
        
        print("\n🚀 Next Steps for Manta Integration:")
        print("1. Copy the call targets file to Manta input directory:")
        print(f"   cp '{manta_file}' $MANTA_DIR_HOME/input/cobol/${{cobol.dictionary.id}}/")
        print("2. Set cobol.call.targets.file property to point to the copied file")
        if unresolved_dynamic:
            print("3. Review and resolve unresolved dynamic calls in the config template")
            print("4. Add resolved entries to the Manta call targets file")
        print("5. Run Manta Data Lineage analysis")
        print("="*70)
        
        # Return appropriate exit code
        return 1 if results.errors else 0
        
    except Exception as e:
        logger.error(f"Scanner failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
