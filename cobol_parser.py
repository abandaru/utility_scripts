#!/usr/bin/env python3
"""
Enterprise COBOL Parser
A comprehensive tool for parsing COBOL source files and analyzing dependencies.

This module provides:
- COBOL source code parsing (.cob, .jcl files)
- Dependency relationship analysis
- Copybook detection
- SQL and CICS command extraction
- Program call analysis
- Extensible architecture with plugin support
"""

import os
import re
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cobol_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class COBOLConstructType(Enum):
    """Enumeration of COBOL construct types."""
    PROGRAM_CALL = "PROGRAM_CALL"
    COPYBOOK = "COPYBOOK" 
    SQL_BLOCK = "SQL_BLOCK"
    CICS_BLOCK = "CICS_BLOCK"
    PROCEDURE = "PROCEDURE"
    SUBPROGRAM = "SUBPROGRAM"
    FILE_CONTROL = "FILE_CONTROL"
    JCL_STEP = "JCL_STEP"


@dataclass
class COBOLConstruct:
    """Represents a COBOL construct found in source code."""
    construct_type: COBOLConstructType
    name: str
    line_number: int
    source_file: str
    content: str = ""
    parameters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramDependency:
    """Represents a dependency relationship between programs."""
    source_program: str
    target_program: str
    dependency_type: str
    line_number: int
    context: str = ""


@dataclass
class ParseResult:
    """Container for parse results."""
    source_file: str
    program_name: Optional[str]
    constructs: List[COBOLConstruct] = field(default_factory=list)
    dependencies: List[ProgramDependency] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class COBOLPatterns:
    """Regular expression patterns for COBOL constructs."""
    
    # Program identification
    PROGRAM_ID = re.compile(
        r'^\s*PROGRAM-ID\.\s+([A-Za-z0-9\-_]+)', 
        re.IGNORECASE | re.MULTILINE
    )
    
    # CALL statements
    STATIC_CALL = re.compile(
        r'^\s*CALL\s+["\']([^"\']+)["\'](?:\s+USING\s+(.*?))?\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    DYNAMIC_CALL = re.compile(
        r'^\s*CALL\s+([A-Za-z0-9\-_]+)(?:\s+USING\s+(.*?))?\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # COPY statements (copybooks)
    COPY_STATEMENT = re.compile(
        r'^\s*COPY\s+([A-Za-z0-9\-_]+)(?:\s+(?:OF|IN)\s+([A-Za-z0-9\-_]+))?\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # SQL blocks
    EXEC_SQL_START = re.compile(
        r'^\s*EXEC\s+SQL\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    EXEC_SQL_END = re.compile(
        r'^\s*END-EXEC\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # CICS blocks
    EXEC_CICS_START = re.compile(
        r'^\s*EXEC\s+CICS\s+(.*?)$',
        re.IGNORECASE | re.MULTILINE
    )
    
    EXEC_CICS_END = re.compile(
        r'^\s*END-EXEC\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Procedures and sections
    PROCEDURE_DIVISION = re.compile(
        r'^\s*PROCEDURE\s+DIVISION(?:\s+USING\s+(.*?))?\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    SECTION_DEFINITION = re.compile(
        r'^\s*([A-Za-z0-9\-_]+)\s+SECTION\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    PARAGRAPH_DEFINITION = re.compile(
        r'^\s*([A-Za-z0-9\-_]+)\.?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # JCL patterns
    JCL_JOB = re.compile(
        r'^//([A-Za-z0-9\-_]+)\s+JOB\s+',
        re.IGNORECASE | re.MULTILINE
    )
    
    JCL_EXEC = re.compile(
        r'^//([A-Za-z0-9\-_]+)\s+EXEC\s+(?:PGM=)?([A-Za-z0-9\-_]+)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # File control
    FILE_CONTROL = re.compile(
        r'^\s*SELECT\s+([A-Za-z0-9\-_]+)\s+ASSIGN\s+TO\s+([^\s]+)',
        re.IGNORECASE | re.MULTILINE
    )


class COBOLParseError(Exception):
    """Custom exception for COBOL parsing errors."""
    
    def __init__(self, message: str, line_number: int = 0, file_path: str = ""):
        self.message = message
        self.line_number = line_number
        self.file_path = file_path
        super().__init__(f"{message} (File: {file_path}, Line: {line_number})")


class COBOLProcessor:
    """Base class for processing specific COBOL constructs."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        """Process lines and return found constructs."""
        raise NotImplementedError("Subclasses must implement process method")


class CallProcessor(COBOLProcessor):
    """Processes CALL statements."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        constructs = []
        
        for line_num, line in enumerate(lines, 1):
            try:
                # Static CALL
                match = COBOLPatterns.STATIC_CALL.match(line.strip())
                if match:
                    program_name = match.group(1)
                    using_clause = match.group(2) if match.group(2) else ""
                    parameters = [p.strip() for p in using_clause.split()] if using_clause else []
                    
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.PROGRAM_CALL,
                        name=program_name,
                        line_number=line_num,
                        source_file=file_path,
                        content=line.strip(),
                        parameters=parameters,
                        metadata={"call_type": "static"}
                    ))
                    continue
                
                # Dynamic CALL
                match = COBOLPatterns.DYNAMIC_CALL.match(line.strip())
                if match:
                    variable_name = match.group(1)
                    using_clause = match.group(2) if match.group(2) else ""
                    parameters = [p.strip() for p in using_clause.split()] if using_clause else []
                    
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.PROGRAM_CALL,
                        name=variable_name,
                        line_number=line_num,
                        source_file=file_path,
                        content=line.strip(),
                        parameters=parameters,
                        metadata={"call_type": "dynamic"}
                    ))
                    
            except Exception as e:
                logger.warning(f"Error processing CALL at line {line_num} in {file_path}: {e}")
                
        return constructs


class CopybookProcessor(COBOLProcessor):
    """Processes COPY statements."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        constructs = []
        
        for line_num, line in enumerate(lines, 1):
            try:
                match = COBOLPatterns.COPY_STATEMENT.match(line.strip())
                if match:
                    copybook_name = match.group(1)
                    library_name = match.group(2) if match.group(2) else ""
                    
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.COPYBOOK,
                        name=copybook_name,
                        line_number=line_num,
                        source_file=file_path,
                        content=line.strip(),
                        metadata={"library": library_name}
                    ))
                    
            except Exception as e:
                logger.warning(f"Error processing COPY at line {line_num} in {file_path}: {e}")
                
        return constructs


class SQLProcessor(COBOLProcessor):
    """Processes EXEC SQL blocks."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        constructs = []
        in_sql_block = False
        sql_content = []
        sql_start_line = 0
        
        for line_num, line in enumerate(lines, 1):
            try:
                stripped_line = line.strip()
                
                if COBOLPatterns.EXEC_SQL_START.match(stripped_line):
                    in_sql_block = True
                    sql_start_line = line_num
                    sql_content = [stripped_line]
                    continue
                
                if in_sql_block:
                    sql_content.append(stripped_line)
                    
                    if COBOLPatterns.EXEC_SQL_END.match(stripped_line):
                        constructs.append(COBOLConstruct(
                            construct_type=COBOLConstructType.SQL_BLOCK,
                            name=f"SQL_BLOCK_{sql_start_line}",
                            line_number=sql_start_line,
                            source_file=file_path,
                            content="\n".join(sql_content)
                        ))
                        in_sql_block = False
                        sql_content = []
                        
            except Exception as e:
                logger.warning(f"Error processing SQL at line {line_num} in {file_path}: {e}")
                
        return constructs


class CICSProcessor(COBOLProcessor):
    """Processes EXEC CICS blocks."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        constructs = []
        
        for line_num, line in enumerate(lines, 1):
            try:
                stripped_line = line.strip()
                
                match = COBOLPatterns.EXEC_CICS_START.match(stripped_line)
                if match:
                    cics_command = match.group(1)
                    
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.CICS_BLOCK,
                        name=f"CICS_{cics_command.split()[0] if cics_command else 'COMMAND'}",
                        line_number=line_num,
                        source_file=file_path,
                        content=stripped_line,
                        metadata={"command": cics_command}
                    ))
                    
            except Exception as e:
                logger.warning(f"Error processing CICS at line {line_num} in {file_path}: {e}")
                
        return constructs


class JCLProcessor(COBOLProcessor):
    """Processes JCL files."""
    
    def process(self, lines: List[str], file_path: str) -> List[COBOLConstruct]:
        constructs = []
        
        for line_num, line in enumerate(lines, 1):
            try:
                # JOB statement
                match = COBOLPatterns.JCL_JOB.match(line)
                if match:
                    job_name = match.group(1)
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.JCL_STEP,
                        name=job_name,
                        line_number=line_num,
                        source_file=file_path,
                        content=line.strip(),
                        metadata={"type": "JOB"}
                    ))
                    continue
                
                # EXEC statement
                match = COBOLPatterns.JCL_EXEC.match(line)
                if match:
                    step_name = match.group(1)
                    program_name = match.group(2)
                    constructs.append(COBOLConstruct(
                        construct_type=COBOLConstructType.JCL_STEP,
                        name=step_name,
                        line_number=line_num,
                        source_file=file_path,
                        content=line.strip(),
                        metadata={"type": "EXEC", "program": program_name}
                    ))
                    
            except Exception as e:
                logger.warning(f"Error processing JCL at line {line_num} in {file_path}: {e}")
                
        return constructs


class COBOLParser:
    """Main COBOL parser class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.processors = {
            'call': CallProcessor(),
            'copybook': CopybookProcessor(),
            'sql': SQLProcessor(),
            'cics': CICSProcessor(),
            'jcl': JCLProcessor()
        }
        self.supported_extensions = {'.cob', '.jcl', '.cbl', '.cpy'}
        
    def add_processor(self, name: str, processor: COBOLProcessor):
        """Add a custom processor for extensibility."""
        self.processors[name] = processor
        logger.info(f"Added custom processor: {name}")
        
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single COBOL file."""
        result = ParseResult(source_file=file_path, program_name=None)
        
        try:
            if not os.path.exists(file_path):
                raise COBOLParseError(f"File not found: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                
            # Extract program name
            result.program_name = self._extract_program_name(lines)
            
            # Process with all processors
            for processor_name, processor in self.processors.items():
                try:
                    if processor_name == 'jcl' and not file_path.endswith('.jcl'):
                        continue
                    if processor_name != 'jcl' and file_path.endswith('.jcl'):
                        continue
                        
                    constructs = processor.process(lines, file_path)
                    result.constructs.extend(constructs)
                    
                except Exception as e:
                    error_msg = f"Error in {processor_name} processor: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
                    
            # Generate dependencies
            result.dependencies = self._generate_dependencies(result)
            
            logger.info(f"Successfully parsed {file_path}: {len(result.constructs)} constructs found")
            
        except Exception as e:
            error_msg = f"Failed to parse {file_path}: {str(e)}"
            result.errors.append(error_msg)
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
        return result
        
    def parse_directory(self, directory_path: str, recursive: bool = True) -> List[ParseResult]:
        """Parse all COBOL files in a directory."""
        results = []
        files_to_process = []
        
        try:
            # Collect files
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_supported_file(file_path):
                            files_to_process.append(file_path)
            else:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and self._is_supported_file(file_path):
                        files_to_process.append(file_path)
                        
            logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process files in parallel
            max_workers = self.config.get('max_workers', 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.parse_file, file_path): file_path 
                    for file_path in files_to_process
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Exception processing {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            
        return results
        
    def _extract_program_name(self, lines: List[str]) -> Optional[str]:
        """Extract program name from PROGRAM-ID statement."""
        for line in lines:
            match = COBOLPatterns.PROGRAM_ID.match(line.strip())
            if match:
                return match.group(1)
        return None
        
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file has supported extension."""
        return Path(file_path).suffix.lower() in self.supported_extensions
        
    def _generate_dependencies(self, result: ParseResult) -> List[ProgramDependency]:
        """Generate dependency relationships from constructs."""
        dependencies = []
        
        for construct in result.constructs:
            if construct.construct_type == COBOLConstructType.PROGRAM_CALL:
                dependencies.append(ProgramDependency(
                    source_program=result.program_name or os.path.basename(result.source_file),
                    target_program=construct.name,
                    dependency_type="CALL",
                    line_number=construct.line_number,
                    context=construct.content
                ))
            elif construct.construct_type == COBOLConstructType.COPYBOOK:
                dependencies.append(ProgramDependency(
                    source_program=result.program_name or os.path.basename(result.source_file),
                    target_program=construct.name,
                    dependency_type="COPYBOOK",
                    line_number=construct.line_number,
                    context=construct.content
                ))
                
        return dependencies


class DependencyAnalyzer:
    """Analyzes dependencies across parsed results."""
    
    def __init__(self, results: List[ParseResult]):
        self.results = results
        self.dependency_graph = defaultdict(set)
        self.reverse_dependency_graph = defaultdict(set)
        self._build_graphs()
        
    def _build_graphs(self):
        """Build dependency graphs."""
        for result in self.results:
            for dep in result.dependencies:
                self.dependency_graph[dep.source_program].add(dep.target_program)
                self.reverse_dependency_graph[dep.target_program].add(dep.source_program)
                
    def get_dependencies(self, program_name: str) -> Set[str]:
        """Get direct dependencies of a program."""
        return self.dependency_graph.get(program_name, set())
        
    def get_dependents(self, program_name: str) -> Set[str]:
        """Get programs that depend on this program."""
        return self.reverse_dependency_graph.get(program_name, set())
        
    def get_transitive_dependencies(self, program_name: str) -> Set[str]:
        """Get all transitive dependencies."""
        visited = set()
        stack = [program_name]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for dep in self.dependency_graph.get(current, set()):
                if dep not in visited:
                    stack.append(dep)
                    
        visited.discard(program_name)
        return visited
        
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                dfs(neighbor, path.copy())
                
            rec_stack.remove(node)
            
        for program in self.dependency_graph:
            if program not in visited:
                dfs(program, [])
                
        return cycles


class ReportGenerator:
    """Generates various reports from parse results."""
    
    def __init__(self, results: List[ParseResult], analyzer: DependencyAnalyzer):
        self.results = results
        self.analyzer = analyzer
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report."""
        total_files = len(self.results)
        total_constructs = sum(len(result.constructs) for result in self.results)
        total_dependencies = sum(len(result.dependencies) for result in self.results)
        total_errors = sum(len(result.errors) for result in self.results)
        
        construct_types = defaultdict(int)
        for result in self.results:
            for construct in result.constructs:
                construct_types[construct.construct_type.value] += 1
                
        return {
            "summary": {
                "total_files_processed": total_files,
                "total_constructs_found": total_constructs,
                "total_dependencies": total_dependencies,
                "total_errors": total_errors,
                "generated_at": datetime.now().isoformat()
            },
            "construct_breakdown": dict(construct_types),
            "files_with_errors": [
                result.source_file for result in self.results if result.errors
            ]
        }
        
    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate a dependency analysis report."""
        all_programs = set()
        for result in self.results:
            if result.program_name:
                all_programs.add(result.program_name)
                
        dependency_metrics = {}
        for program in all_programs:
            dependencies = self.analyzer.get_dependencies(program)
            dependents = self.analyzer.get_dependents(program)
            
            dependency_metrics[program] = {
                "outgoing_dependencies": len(dependencies),
                "incoming_dependencies": len(dependents),
                "fan_out": list(dependencies),
                "fan_in": list(dependents)
            }
            
        circular_deps = self.analyzer.find_circular_dependencies()
        
        return {
            "dependency_metrics": dependency_metrics,
            "circular_dependencies": circular_deps,
            "isolated_programs": [
                prog for prog, metrics in dependency_metrics.items()
                if metrics["outgoing_dependencies"] == 0 and metrics["incoming_dependencies"] == 0
            ]
        }
        
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed report with all information."""
        detailed_results = []
        
        for result in self.results:
            constructs_by_type = defaultdict(list)
            for construct in result.constructs:
                constructs_by_type[construct.construct_type.value].append(asdict(construct))
                
            detailed_results.append({
                "file_path": result.source_file,
                "program_name": result.program_name,
                "constructs_by_type": dict(constructs_by_type),
                "dependencies": [asdict(dep) for dep in result.dependencies],
                "errors": result.errors,
                "warnings": result.warnings
            })
            
        return {
            "detailed_results": detailed_results,
            "summary": self.generate_summary_report(),
            "dependency_analysis": self.generate_dependency_report()
        }
    
    def generate_csv_relationships_report(self, output_file: str) -> None:
        """Generate CSV report of all program relationships."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Source Program', 'Target Program', 'Relationship Type',
                    'Source File', 'Line Number', 'Context', 'Call Type',
                    'Library', 'Parameters Count', 'Content Preview'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    source_program = result.program_name or Path(result.source_file).stem
                    
                    for construct in result.constructs:
                        # Program calls
                        if construct.construct_type == COBOLConstructType.PROGRAM_CALL:
                            writer.writerow({
                                'Source Program': source_program,
                                'Target Program': construct.name,
                                'Relationship Type': 'PROGRAM_CALL',
                                'Source File': result.source_file,
                                'Line Number': construct.line_number,
                                'Context': construct.content[:100] + '...' if len(construct.content) > 100 else construct.content,
                                'Call Type': construct.metadata.get('call_type', ''),
                                'Library': '',
                                'Parameters Count': len(construct.parameters),
                                'Content Preview': ' '.join(construct.parameters[:3])
                            })
                        
                        # Copybook relationships
                        elif construct.construct_type == COBOLConstructType.COPYBOOK:
                            writer.writerow({
                                'Source Program': source_program,
                                'Target Program': construct.name,
                                'Relationship Type': 'COPYBOOK_INCLUDE',
                                'Source File': result.source_file,
                                'Line Number': construct.line_number,
                                'Context': construct.content,
                                'Call Type': '',
                                'Library': construct.metadata.get('library', ''),
                                'Parameters Count': 0,
                                'Content Preview': construct.metadata.get('library', '')
                            })
                        
                        # SQL relationships
                        elif construct.construct_type == COBOLConstructType.SQL_BLOCK:
                            writer.writerow({
                                'Source Program': source_program,
                                'Target Program': construct.name,
                                'Relationship Type': 'SQL_BLOCK',
                                'Source File': result.source_file,
                                'Line Number': construct.line_number,
                                'Context': construct.content[:100] + '...' if len(construct.content) > 100 else construct.content,
                                'Call Type': '',
                                'Library': '',
                                'Parameters Count': 0,
                                'Content Preview': self._extract_sql_tables(construct.content)
                            })
                        
                        # CICS relationships
                        elif construct.construct_type == COBOLConstructType.CICS_BLOCK:
                            writer.writerow({
                                'Source Program': source_program,
                                'Target Program': construct.name,
                                'Relationship Type': 'CICS_COMMAND',
                                'Source File': result.source_file,
                                'Line Number': construct.line_number,
                                'Context': construct.content,
                                'Call Type': '',
                                'Library': '',
                                'Parameters Count': 0,
                                'Content Preview': construct.metadata.get('command', '')[:50]
                            })
                        
                        # JCL relationships
                        elif construct.construct_type == COBOLConstructType.JCL_STEP:
                            if 'program' in construct.metadata:
                                writer.writerow({
                                    'Source Program': source_program,
                                    'Target Program': construct.metadata['program'],
                                    'Relationship Type': 'JCL_EXECUTION',
                                    'Source File': result.source_file,
                                    'Line Number': construct.line_number,
                                    'Context': construct.content,
                                    'Call Type': construct.metadata.get('type', ''),
                                    'Library': '',
                                    'Parameters Count': 0,
                                    'Content Preview': construct.name
                                })
            
            logger.info(f"CSV relationships report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV relationships report: {e}")
            raise
    
    def generate_csv_dependency_analysis(self, output_file: str) -> None:
        """Generate CSV report of dependency analysis metrics."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Program Name', 'Source File', 'Outgoing Dependencies Count',
                    'Incoming Dependencies Count', 'Fan Out Programs', 'Fan In Programs',
                    'Transitive Dependencies Count', 'Max Dependency Depth',
                    'Has Circular Dependencies', 'Is Isolated', 'Program Type'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Get all programs
                all_programs = {}
                for result in self.results:
                    if result.program_name:
                        all_programs[result.program_name] = result.source_file
                
                # Find circular dependencies once
                circular_deps = self.analyzer.find_circular_dependencies()
                programs_in_cycles = set()
                for cycle in circular_deps:
                    programs_in_cycles.update(cycle)
                
                for program_name, source_file in all_programs.items():
                    dependencies = self.analyzer.get_dependencies(program_name)
                    dependents = self.analyzer.get_dependents(program_name)
                    transitive_deps = self.analyzer.get_transitive_dependencies(program_name)
                    
                    # Determine program type based on file extension
                    file_ext = Path(source_file).suffix.lower()
                    if file_ext == '.jcl':
                        program_type = 'JCL'
                    elif file_ext in ['.cpy']:
                        program_type = 'COPYBOOK'
                    else:
                        program_type = 'COBOL_PROGRAM'
                    
                    writer.writerow({
                        'Program Name': program_name,
                        'Source File': source_file,
                        'Outgoing Dependencies Count': len(dependencies),
                        'Incoming Dependencies Count': len(dependents),
                        'Fan Out Programs': '; '.join(sorted(dependencies)) if dependencies else '',
                        'Fan In Programs': '; '.join(sorted(dependents)) if dependents else '',
                        'Transitive Dependencies Count': len(transitive_deps),
                        'Max Dependency Depth': self._calculate_max_depth(program_name),
                        'Has Circular Dependencies': 'Yes' if program_name in programs_in_cycles else 'No',
                        'Is Isolated': 'Yes' if len(dependencies) == 0 and len(dependents) == 0 else 'No',
                        'Program Type': program_type
                    })
            
            logger.info(f"CSV dependency analysis report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV dependency analysis: {e}")
            raise
    
    def generate_csv_constructs_summary(self, output_file: str) -> None:
        """Generate CSV summary of all constructs found."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Source File', 'Program Name', 'Construct Type', 'Construct Name',
                    'Line Number', 'Content', 'Metadata', 'Parameters'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    for construct in result.constructs:
                        writer.writerow({
                            'Source File': result.source_file,
                            'Program Name': result.program_name or '',
                            'Construct Type': construct.construct_type.value,
                            'Construct Name': construct.name,
                            'Line Number': construct.line_number,
                            'Content': construct.content[:200] + '...' if len(construct.content) > 200 else construct.content,
                            'Metadata': json.dumps(construct.metadata) if construct.metadata else '',
                            'Parameters': '; '.join(construct.parameters) if construct.parameters else ''
                        })
            
            logger.info(f"CSV constructs summary report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV constructs summary: {e}")
            raise
    
    def generate_csv_circular_dependencies(self, output_file: str) -> None:
        """Generate CSV report of circular dependencies."""
        try:
            circular_deps = self.analyzer.find_circular_dependencies()
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Cycle ID', 'Cycle Length', 'Programs in Cycle', 'Cycle Path',
                    'Risk Level', 'First Program', 'Last Program'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, cycle in enumerate(circular_deps, 1):
                    risk_level = 'HIGH' if len(cycle) <= 3 else 'MEDIUM' if len(cycle) <= 5 else 'LOW'
                    
                    writer.writerow({
                        'Cycle ID': f"CYCLE_{i:03d}",
                        'Cycle Length': len(cycle),
                        'Programs in Cycle': '; '.join(cycle),
                        'Cycle Path': ' → '.join(cycle + [cycle[0]]),
                        'Risk Level': risk_level,
                        'First Program': cycle[0] if cycle else '',
                        'Last Program': cycle[-1] if cycle else ''
                    })
            
            logger.info(f"CSV circular dependencies report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV circular dependencies report: {e}")
            raise
    
    def generate_csv_program_metrics(self, output_file: str) -> None:
        """Generate CSV report with detailed program metrics."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Program Name', 'Source File', 'File Size (KB)', 'Total Lines',
                    'Total Constructs', 'Call Statements', 'Copybooks Used',
                    'SQL Blocks', 'CICS Commands', 'Errors Count', 'Warnings Count',
                    'Complexity Score', 'Last Modified'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    # Count constructs by type
                    construct_counts = defaultdict(int)
                    for construct in result.constructs:
                        construct_counts[construct.construct_type] += 1
                    
                    # Calculate file metrics
                    try:
                        file_stat = os.stat(result.source_file)
                        file_size_kb = file_stat.st_size / 1024
                        last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        
                        # Count total lines
                        with open(result.source_file, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines = sum(1 for _ in f)
                    except:
                        file_size_kb = 0
                        total_lines = 0
                        last_modified = ''
                    
                    # Calculate complexity score (simple heuristic)
                    complexity_score = (
                        construct_counts[COBOLConstructType.PROGRAM_CALL] * 2 +
                        construct_counts[COBOLConstructType.SQL_BLOCK] * 3 +
                        construct_counts[COBOLConstructType.CICS_BLOCK] * 3 +
                        construct_counts[COBOLConstructType.COPYBOOK] * 1 +
                        len(result.errors) * 5
                    )
                    
                    writer.writerow({
                        'Program Name': result.program_name or Path(result.source_file).stem,
                        'Source File': result.source_file,
                        'File Size (KB)': f"{file_size_kb:.2f}",
                        'Total Lines': total_lines,
                        'Total Constructs': len(result.constructs),
                        'Call Statements': construct_counts[COBOLConstructType.PROGRAM_CALL],
                        'Copybooks Used': construct_counts[COBOLConstructType.COPYBOOK],
                        'SQL Blocks': construct_counts[COBOLConstructType.SQL_BLOCK],
                        'CICS Commands': construct_counts[COBOLConstructType.CICS_BLOCK],
                        'Errors Count': len(result.errors),
                        'Warnings Count': len(result.warnings),
                        'Complexity Score': complexity_score,
                        'Last Modified': last_modified
                    })
            
            logger.info(f"CSV program metrics report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV program metrics: {e}")
            raise
    
    def generate_all_csv_reports(self, output_dir: str, base_filename: str = "cobol_analysis") -> Dict[str, str]:
        """Generate all CSV reports in the specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_files = {
            'relationships': output_path / f"{base_filename}_relationships_{timestamp}.csv",
            'dependencies': output_path / f"{base_filename}_dependencies_{timestamp}.csv",
            'constructs': output_path / f"{base_filename}_constructs_{timestamp}.csv",
            'circular_deps': output_path / f"{base_filename}_circular_deps_{timestamp}.csv",
            'metrics': output_path / f"{base_filename}_metrics_{timestamp}.csv"
        }
        
        try:
            self.generate_csv_relationships_report(str(csv_files['relationships']))
            self.generate_csv_dependency_analysis(str(csv_files['dependencies']))
            self.generate_csv_constructs_summary(str(csv_files['constructs']))
            self.generate_csv_circular_dependencies(str(csv_files['circular_deps']))
            self.generate_csv_program_metrics(str(csv_files['metrics']))
            
            logger.info(f"All CSV reports generated in: {output_dir}")
            return {k: str(v) for k, v in csv_files.items()}
            
        except Exception as e:
            logger.error(f"Error generating CSV reports: {e}")
            raise
    
    def _extract_sql_tables(self, sql_content: str) -> str:
        """Extract table names from SQL content."""
        try:
            # Simple regex to find table names after FROM and JOIN
            table_pattern = re.compile(r'\b(?:FROM|JOIN)\s+([A-Za-z0-9_]+)', re.IGNORECASE)
            tables = table_pattern.findall(sql_content)
            return '; '.join(sorted(set(tables))) if tables else ''
        except:
            return ''
    
    def _calculate_max_depth(self, program_name: str) -> int:
        """Calculate maximum dependency depth for a program."""
        try:
            visited = set()
            
            def dfs(prog, depth):
                if prog in visited:
                    return depth
                visited.add(prog)
                
                max_child_depth = depth
                for dep in self.analyzer.get_dependencies(prog):
                    child_depth = dfs(dep, depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
                
                return max_child_depth
            
            return dfs(program_name, 0)
        except:
            return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enterprise COBOL Parser")
    parser.add_argument("path", help="Path to COBOL file or directory")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--csv-output", help="Output directory for CSV reports")
    parser.add_argument("--csv-name", default="cobol_analysis", 
                       help="Base filename for CSV reports")
    parser.add_argument("--csv-type", choices=[
        "all", "relationships", "dependencies", "constructs", 
        "circular", "metrics"
    ], default="all", help="Type of CSV report to generate")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Recursively process directories")
    parser.add_argument("--report-type", choices=["summary", "dependency", "detailed"], 
                       default="detailed", help="Type of JSON report to generate")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum number of worker threads")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        config = {
            "max_workers": args.max_workers
        }
        
        cobol_parser = COBOLParser(config)
        
        # Parse files
        if os.path.isfile(args.path):
            results = [cobol_parser.parse_file(args.path)]
        elif os.path.isdir(args.path):
            results = cobol_parser.parse_directory(args.path, args.recursive)
        else:
            raise ValueError(f"Invalid path: {args.path}")
            
        # Analyze dependencies
        analyzer = DependencyAnalyzer(results)
        
        # Generate report
        report_generator = ReportGenerator(results, analyzer)
        
        # Generate JSON report if requested
        if args.output or not args.csv_output:
            if args.report_type == "summary":
                report = report_generator.generate_summary_report()
            elif args.report_type == "dependency":
                report = report_generator.generate_dependency_report()
            else:
                report = report_generator.generate_detailed_report()
                
            # Output JSON results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"JSON report written to {args.output}")
            elif not args.csv_output:
                print(json.dumps(report, indent=2))
        
        # Generate CSV reports if requested
        if args.csv_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_output_dir = Path(args.csv_output)
            csv_output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.csv_type == "all":
                csv_files = report_generator.generate_all_csv_reports(
                    str(csv_output_dir), args.csv_name
                )
                print("Generated CSV reports:")
                for report_type, file_path in csv_files.items():
                    print(f"  {report_type.title()}: {file_path}")
                    
            else:
                # Generate specific CSV report
                output_file = csv_output_dir / f"{args.csv_name}_{args.csv_type}_{timestamp}.csv"
                
                if args.csv_type == "relationships":
                    report_generator.generate_csv_relationships_report(str(output_file))
                elif args.csv_type == "dependencies":
                    report_generator.generate_csv_dependency_analysis(str(output_file))
                elif args.csv_type == "constructs":
                    report_generator.generate_csv_constructs_summary(str(output_file))
                elif args.csv_type == "circular":
                    report_generator.generate_csv_circular_dependencies(str(output_file))
                elif args.csv_type == "metrics":
                    report_generator.generate_csv_program_metrics(str(output_file))
                
                print(f"Generated CSV report: {output_file}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
