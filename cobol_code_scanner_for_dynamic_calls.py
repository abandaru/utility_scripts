#!/usr/bin/env python3
"""
COBOL Caller-Callee Scanner for Manta Data Lineage
==================================================

This script scans COBOL files to identify caller-callee relationships between
main programs and sub-programs, generating output compatible with IBM Manta
Data Lineage scanner.

Features:
- Recursive directory scanning
- COBOL program identification
- CALL statement parsing (static and dynamic)
- Variable-based call tracking
- Manta-compatible output format
- Comprehensive error handling and logging

Author: Adi Bandaru
Version: 1.0.0
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
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # COBOL file extensions
        self.cobol_extensions = {'.cbl', '.cob', '.cobol', '.cpy', '.copy'}
        
        # Regex patterns for COBOL parsing
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
                r'^\s*CALL\s+([A-Z0-9\-_]+)\s*(?:USING|$)',
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
        
        # Variable to program name mappings
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
        """Parse a COBOL file to extract program information"""
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
        
        # Build variable to program mappings from MOVE statements
        move_matches = self.patterns['move_statement'].findall(content)
        for program_name, variable_name in move_matches:
            full_key = f"{program_id}.{variable_name}"
            self.variable_mappings[full_key] = program_name
        
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
            
            # Check for static CALL statements
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
            
            # Check for dynamic CALL statements
            dynamic_match = self.patterns['call_dynamic'].search(line)
            if dynamic_match and not self.patterns['call_static'].search(line):
                variable_name = dynamic_match.group(1)
                
                # Try to resolve variable to program name
                full_key = f"{caller_program}.{variable_name}"
                callee_program = self.variable_mappings.get(full_key)
                
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
        
        Returns path to generated file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate call targets file for Manta
        call_targets_file = self.output_dir / f"cobol_call_targets_{timestamp}.csv"
        
        with open(call_targets_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Header as required by Manta
            writer.writerow([
                "Calling program name",
                "Calling variable name", 
                "Target program name"
            ])
            
            # Write relationships
            for rel in results.relationships:
                if rel.call_type == 'dynamic' and rel.callee_program:
                    writer.writerow([
                        rel.caller_program,
                        rel.call_variable or "",
                        rel.callee_program
                    ])
                elif rel.call_type == 'static' and rel.callee_program:
                    writer.writerow([
                        rel.caller_program,
                        "",  # No variable for static calls
                        rel.callee_program
                    ])
        
        logger.info(f"Manta call targets file generated: {call_targets_file}")
        return str(call_targets_file)

    def export_detailed_report(self, results: ScanResults) -> str:
        """Export detailed analysis report in JSON format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"cobol_analysis_report_{timestamp}.json"
        
        report_data = {
            'scan_metadata': {
                'timestamp': timestamp,
                'statistics': results.statistics
            },
            'programs': [asdict(prog) for prog in results.programs],
            'relationships': [asdict(rel) for rel in results.relationships],
            'errors': results.errors,
            'variable_mappings': self.variable_mappings
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed report generated: {report_file}")
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

def main():
    """Main entry point for the scanner"""
    parser = argparse.ArgumentParser(
        description='COBOL Caller-Callee Scanner for Manta Data Lineage'
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
        
        # Generate outputs
        manta_file = scanner.export_manta_format(results)
        report_file = scanner.export_detailed_report(results)
        summary_file = scanner.export_program_summary(results)
        
        # Print summary
        print("\n" + "="*60)
        print("COBOL SCAN RESULTS SUMMARY")
        print("="*60)
        print(f"Files scanned: {results.statistics['files_scanned']}")
        print(f"Programs found: {results.statistics['programs_found']}")
        print(f"Static calls: {results.statistics['static_calls']}")
        print(f"Dynamic calls: {results.statistics['dynamic_calls']}")
        print(f"Copy statements: {results.statistics['copy_statements']}")
        print(f"Errors: {results.statistics['errors']}")
        print("\nGenerated files:")
        print(f"- Manta call targets: {manta_file}")
        print(f"- Detailed report: {report_file}")
        print(f"- Program summary: {summary_file}")
        
        if results.errors:
            print(f"\nErrors encountered: {len(results.errors)}")
            for error in results.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results.errors) > 5:
                print(f"  ... and {len(results.errors) - 5} more errors")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Scanner failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
