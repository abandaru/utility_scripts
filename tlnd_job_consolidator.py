#!/usr/bin/env python3
"""
Corrected Talend Job Consolidator for Data Lineage Analysis

- Proper XML namespace handling
- Robust joblet detection
- Better error handling
- Improved connection reference updates

Author: Adi Bandaru
Version: 1.0 
"""

import os
import xml.etree.ElementTree as ET
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import shutil
import copy


@dataclass
class JobComponent:
    """Represents a Talend job component"""
    component_name: str
    component_type: str
    unique_name: str
    properties: Dict[str, str] = field(default_factory=dict)
    xml_element: Optional[ET.Element] = None
    position: Tuple[int, int] = (0, 0)


@dataclass
class JobConnection:
    """Represents a connection between components"""
    connector_name: str
    source: str
    target: str
    label: str = ""
    xml_element: Optional[ET.Element] = None


@dataclass
class JobFile:
    """Represents a Talend job file"""
    file_path: Path
    job_name: str
    job_type: str  # 'main', 'joblet', 'shared'
    components: List[JobComponent] = field(default_factory=list)
    connections: List[JobConnection] = field(default_factory=list)
    joblet_references: Set[str] = field(default_factory=set)
    xml_root: Optional[ET.Element] = None
    namespaces: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConsolidationSummary:
    """Summary of the consolidation process"""
    total_files_processed: int = 0
    main_jobs_found: int = 0
    joblets_found: int = 0
    shared_jobs_found: int = 0
    consolidated_jobs_created: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    joblet_merges: Dict[str, List[str]] = field(default_factory=dict)
    processing_time: float = 0.0


class TalendJobConsolidator:
    """
    Corrected main class for consolidating Talend job files
    """
    
    # Common Talend namespaces
    TALEND_NAMESPACES = {
        'talendfile': 'platform:/resource/org.talend.model/model/TalendFile.xsd',
        'xmi': 'http://www.omg.org/XMI',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    # Known joblet component patterns
    JOBLET_PATTERNS = [
        r'.*joblet.*',
        r'.*Joblet.*', 
        r'tJoblet.*',
        r'.*_joblet',
        r'.*subjob.*'
    ]
    
    # Properties that may contain joblet references
    JOBLET_PROPERTY_NAMES = [
        'SELECTED_JOB_NAME',
        'JOBLET_NAME', 
        'PROCESS_TYPE_PROCESS',
        'PROCESS',
        'JOB_NAME',
        'SUBJOB_NAME'
    ]
    
    def __init__(self, input_folders: List[str], output_folder: str):
        """
        Initialize the consolidator
        
        Args:
            input_folders: List of folder paths containing Talend job files
            output_folder: Path where consolidated job files will be saved
        """
        self.input_folders = [Path(folder) for folder in input_folders]
        self.output_folder = Path(output_folder)
        self.job_files: Dict[str, JobFile] = {}
        self.joblets: Dict[str, JobFile] = {}
        self.main_jobs: Dict[str, JobFile] = {}
        self.shared_jobs: Dict[str, JobFile] = {}
        self.summary = ConsolidationSummary()
        
        # Setup logging
        self._setup_logging()
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('talend_consolidator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_folders(self) -> None:
        """
        Scan input folders for Talend job files
        """
        self.logger.info("Starting folder scan...")
        start_time = datetime.now()
        
        for folder in self.input_folders:
            if not folder.exists():
                error_msg = f"Input folder does not exist: {folder}"
                self.logger.error(error_msg)
                self.summary.errors.append(error_msg)
                continue
            
            self.logger.info(f"Scanning folder: {folder}")
            self._scan_folder_recursive(folder)
        
        self.summary.processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Folder scan completed in {self.summary.processing_time:.2f} seconds")
    
    def _scan_folder_recursive(self, folder: Path) -> None:
        """
        Recursively scan folder for Talend files
        
        Args:
            folder: Folder to scan
        """
        try:
            # Look for .item files (Talend job definitions)
            for file_path in folder.rglob("*.item"):
                try:
                    self._process_job_file(file_path)
                    self.summary.total_files_processed += 1
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    self.summary.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error scanning folder {folder}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.summary.errors.append(error_msg)
    
    def _process_job_file(self, file_path: Path) -> None:
        """
        Process a single Talend job file
        
        Args:
            file_path: Path to the job file
        """
        self.logger.debug(f"Processing file: {file_path}")
        
        try:
            # Parse XML file with namespace support
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract namespaces
            namespaces = self._extract_namespaces(root)
            
            # Determine job type and name
            job_type = self._determine_job_type(root, file_path)
            job_name = self._extract_job_name(root, file_path)
            
            if not job_name:
                warning_msg = f"Could not determine job name for {file_path}, using filename"
                self.logger.warning(warning_msg)
                self.summary.warnings.append(warning_msg)
                job_name = file_path.stem
            
            # Create JobFile object
            job_file = JobFile(
                file_path=file_path,
                job_name=job_name,
                job_type=job_type,
                xml_root=root,
                namespaces=namespaces
            )
            
            # Extract components and connections
            self._extract_components(job_file)
            self._extract_connections(job_file)
            
            # Store in appropriate collection
            self._categorize_job_file(job_file)
            
            self.logger.debug(f"Processed {job_type}: {job_name} with {len(job_file.components)} components")
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error in {file_path}: {str(e)}"
            self.logger.error(error_msg)
            self.summary.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error processing {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.summary.errors.append(error_msg)
    
    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """
        Extract XML namespaces from root element
        
        Args:
            root: XML root element
            
        Returns:
            Dictionary of namespace prefixes to URIs
        """
        namespaces = {}
        
        # Get namespaces from root element
        for prefix, uri in root.attrib.items():
            if prefix.startswith('xmlns:'):
                namespace_prefix = prefix[6:]  # Remove 'xmlns:' prefix
                namespaces[namespace_prefix] = uri
            elif prefix == 'xmlns':
                namespaces[''] = uri  # Default namespace
        
        # Add common Talend namespaces if not present
        for prefix, uri in self.TALEND_NAMESPACES.items():
            if prefix not in namespaces:
                namespaces[prefix] = uri
        
        return namespaces
    
    def _determine_job_type(self, root: ET.Element, file_path: Path) -> str:
        """
        Determine the type of Talend job with improved logic
        
        Args:
            root: XML root element
            file_path: Path to the file
            
        Returns:
            Job type ('main', 'joblet', 'shared')
        """
        # Check file path patterns first
        file_path_str = str(file_path).lower()
        parent_dir = file_path.parent.name.lower()
        filename = file_path.stem.lower()
        
        # Strong indicators from path
        if any(indicator in file_path_str for indicator in ['joblet', 'subjob']):
            return 'joblet'
        if any(indicator in file_path_str for indicator in ['shared', 'utility', 'common']):
            return 'shared'
        
        # Check XML content
        try:
            # Check root element attributes
            job_type_attr = root.get('jobType', '').lower()
            if job_type_attr == 'joblet':
                return 'joblet'
            
            # Look for joblet-specific elements with namespace support
            joblet_indicators = [
                './/elementParameter[@name="JOBLET_ICON"]',
                './/elementParameter[@field="ICON"]',
                './/*[@jobType="Joblet"]'
            ]
            
            for indicator in joblet_indicators:
                if root.find(indicator) is not None:
                    return 'joblet'
            
            # Check for shared job indicators
            if 'shared' in filename or 'utility' in filename or 'common' in filename:
                return 'shared'
            
        except Exception as e:
            self.logger.debug(f"Error checking XML content for job type: {e}")
        
        # Default to main job
        return 'main'
    
    def _extract_job_name(self, root: ET.Element, file_path: Path) -> Optional[str]:
        """
        Extract job name from XML or file path with improved logic
        
        Args:
            root: XML root element
            file_path: Path to the file
            
        Returns:
            Job name or None if not found
        """
        # Try multiple attributes for job name
        name_attributes = ['name', 'label', 'id']
        
        for attr in name_attributes:
            name = root.get(attr)
            if name and name.strip():
                return name.strip()
        
        # Try to find name in child elements
        try:
            process_element = root.find('.//process')
            if process_element is not None:
                for attr in name_attributes:
                    name = process_element.get(attr)
                    if name and name.strip():
                        return name.strip()
        except Exception:
            pass
        
        # Fallback to filename without extension
        return file_path.stem
    
    def _extract_components(self, job_file: JobFile) -> None:
        """
        Extract components from job XML with improved parsing
        
        Args:
            job_file: JobFile object to populate
        """
        root = job_file.xml_root
        
        # Find all node elements (components) with namespace support
        node_selectors = [
            './/node',
            './/talendfile:node',
            './/*[@componentName]'
        ]
        
        nodes_found = []
        for selector in node_selectors:
            try:
                nodes = root.findall(selector)
                if nodes:
                    nodes_found.extend(nodes)
                    break  # Use first successful selector
            except Exception as e:
                self.logger.debug(f"Selector {selector} failed: {e}")
        
        for node in nodes_found:
            try:
                component_name = node.get('componentName', '')
                component_type = node.get('componentVersion', '')
                unique_name = node.get('uniqueName', '')
                
                # Extract position if available
                pos_x = int(node.get('posX', 0))
                pos_y = int(node.get('posY', 0))
                
                # Extract properties with improved parsing
                properties = self._extract_element_parameters(node)
                
                # Create component
                component = JobComponent(
                    component_name=component_name,
                    component_type=component_type,
                    unique_name=unique_name,
                    properties=properties,
                    xml_element=node,
                    position=(pos_x, pos_y)
                )
                
                job_file.components.append(component)
                
                # Check if this is a joblet reference with improved detection
                if self._is_joblet_component(component):
                    joblet_name = self._extract_joblet_name(component)
                    if joblet_name:
                        job_file.joblet_references.add(joblet_name)
                        self.logger.debug(f"Found joblet reference: {joblet_name}")
                
            except Exception as e:
                self.logger.warning(f"Error extracting component from node: {e}")
    
    def _extract_connections(self, job_file: JobFile) -> None:
        """
        Extract connections from job XML
        
        Args:
            job_file: JobFile object to populate
        """
        root = job_file.xml_root
        
        # Find all connection elements
        connection_selectors = [
            './/connection',
            './/talendfile:connection'
        ]
        
        connections_found = []
        for selector in connection_selectors:
            try:
                connections = root.findall(selector)
                if connections:
                    connections_found.extend(connections)
                    break
            except Exception as e:
                self.logger.debug(f"Connection selector {selector} failed: {e}")
        
        for conn in connections_found:
            try:
                connection = JobConnection(
                    connector_name=conn.get('connectorName', ''),
                    source=conn.get('source', ''),
                    target=conn.get('target', ''),
                    label=conn.get('label', ''),
                    xml_element=conn
                )
                job_file.connections.append(connection)
            except Exception as e:
                self.logger.warning(f"Error extracting connection: {e}")
    
    def _extract_element_parameters(self, node: ET.Element) -> Dict[str, str]:
        """
        Extract elementParameter values from a node
        
        Args:
            node: XML node element
            
        Returns:
            Dictionary of parameter names to values
        """
        properties = {}
        
        try:
            # Find all elementParameter elements
            for param in node.findall('.//elementParameter'):
                param_name = param.get('name', '')
                param_value = param.get('value', '')
                
                if param_name and param_value:
                    # Clean up the value (remove quotes)
                    cleaned_value = self._clean_parameter_value(param_value)
                    properties[param_name] = cleaned_value
        except Exception as e:
            self.logger.debug(f"Error extracting parameters: {e}")
        
        return properties
    
    def _clean_parameter_value(self, value: str) -> str:
        """
        Clean parameter value by removing quotes and escapes
        
        Args:
            value: Raw parameter value
            
        Returns:
            Cleaned value
        """
        if not value:
            return value
        
        # Remove surrounding quotes
        cleaned = value.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        
        return cleaned
    
    def _is_joblet_component(self, component: JobComponent) -> bool:
        """
        Check if component is a joblet reference with improved detection
        
        Args:
            component: Component to check
            
        Returns:
            True if component is a joblet reference
        """
        # Check component name against patterns
        component_name_lower = component.component_name.lower()
        
        for pattern in self.JOBLET_PATTERNS:
            if re.match(pattern, component_name_lower):
                return True
        
        # Check if any properties contain joblet references
        for prop_name in self.JOBLET_PROPERTY_NAMES:
            if prop_name in component.properties:
                # Additional validation that the property value looks like a job name
                value = component.properties[prop_name]
                if value and len(value) > 0 and not value.startswith('/'):
                    return True
        
        return False
    
    def _extract_joblet_name(self, component: JobComponent) -> Optional[str]:
        """
        Extract joblet name from component properties with improved logic
        
        Args:
            component: Joblet component
            
        Returns:
            Joblet name if found
        """
        # Try each property name in order of preference
        for prop_name in self.JOBLET_PROPERTY_NAMES:
            if prop_name in component.properties:
                value = component.properties[prop_name]
                cleaned_value = self._clean_parameter_value(value)
                
                # Validate that this looks like a joblet name
                if cleaned_value and self._is_valid_joblet_name(cleaned_value):
                    return cleaned_value
        
        # If no property found, try to infer from component name
        component_name = component.component_name
        if 'joblet' in component_name.lower():
            # Remove common suffixes/prefixes
            cleaned_name = re.sub(r'(^t|_\d+$)', '', component_name)
            if self._is_valid_joblet_name(cleaned_name):
                return cleaned_name
        
        return None
    
    def _is_valid_joblet_name(self, name: str) -> bool:
        """
        Validate that a string looks like a valid joblet name
        
        Args:
            name: Potential joblet name
            
        Returns:
            True if valid joblet name
        """
        if not name or len(name) < 2:
            return False
        
        # Should not be a file path
        if '/' in name or '\\' in name:
            return False
        
        # Should not be a URL
        if name.startswith('http'):
            return False
        
        # Should contain letters
        if not re.search(r'[a-zA-Z]', name):
            return False
        
        return True
    
    def _categorize_job_file(self, job_file: JobFile) -> None:
        """
        Categorize job file into appropriate collection
        
        Args:
            job_file: JobFile to categorize
        """
        self.job_files[job_file.job_name] = job_file
        
        if job_file.job_type == 'main':
            self.main_jobs[job_file.job_name] = job_file
            self.summary.main_jobs_found += 1
        elif job_file.job_type == 'joblet':
            self.joblets[job_file.job_name] = job_file
            self.summary.joblets_found += 1
        elif job_file.job_type == 'shared':
            self.shared_jobs[job_file.job_name] = job_file
            self.summary.shared_jobs_found += 1
    
    def consolidate_jobs(self) -> None:
        """
        Consolidate main jobs with their joblets
        """
        self.logger.info("Starting job consolidation...")
        
        for job_name, main_job in self.main_jobs.items():
            try:
                self.logger.info(f"Consolidating job: {job_name}")
                
                # Create consolidated job
                consolidated_job = self._create_consolidated_job(main_job)
                
                # Save consolidated job
                output_file = self.output_folder / f"{job_name}_consolidated.item"
                self._save_consolidated_job(consolidated_job, output_file)
                
                self.summary.consolidated_jobs_created += 1
                self.logger.info(f"Created consolidated job: {output_file}")
                
            except Exception as e:
                error_msg = f"Error consolidating job {job_name}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.summary.errors.append(error_msg)
    
    def _create_consolidated_job(self, main_job: JobFile) -> ET.Element:
        """
        Create consolidated job by merging joblets with improved logic
        
        Args:
            main_job: Main job to consolidate
            
        Returns:
            Consolidated XML root element
        """
        # Create a deep copy of the main job XML
        consolidated_root = copy.deepcopy(main_job.xml_root)
        
        # Track merged joblets for summary
        merged_joblets = []
        
        # Process each joblet reference
        for joblet_ref in main_job.joblet_references:
            if joblet_ref in self.joblets:
                joblet = self.joblets[joblet_ref]
                self.logger.debug(f"Merging joblet: {joblet_ref}")
                
                # Merge joblet into consolidated job
                self._merge_joblet_into_job(consolidated_root, joblet, main_job.job_name)
                merged_joblets.append(joblet_ref)
            else:
                warning_msg = f"Joblet not found: {joblet_ref} (referenced in {main_job.job_name})"
                self.logger.warning(warning_msg)
                self.summary.warnings.append(warning_msg)
        
        # Update summary
        if merged_joblets:
            self.summary.joblet_merges[main_job.job_name] = merged_joblets
        
        return consolidated_root
    
    def _merge_joblet_into_job(self, main_root: ET.Element, joblet: JobFile, main_job_name: str) -> None:
        """
        Merge joblet components into main job with improved logic
        
        Args:
            main_root: Main job XML root
            joblet: Joblet to merge
            main_job_name: Name of main job for unique naming
        """
        # Find the process element in main job
        process_element = main_root.find('.//process')
        if process_element is None:
            # Try with namespace
            for ns_prefix in ['talendfile', '']:
                if ns_prefix:
                    process_element = main_root.find(f'.//{ns_prefix}:process')
                else:
                    process_element = main_root.find('.//process')
                if process_element is not None:
                    break
        
        if process_element is None:
            # Create process element if it doesn't exist
            process_element = ET.SubElement(main_root, 'process')
        
        # Get joblet process element
        joblet_process = joblet.xml_root.find('.//process')
        if joblet_process is None:
            self.logger.warning(f"No process element found in joblet {joblet.job_name}")
            return
        
        # Merge components
        self._merge_components(process_element, joblet_process, joblet.job_name)
        
        # Merge connections
        self._merge_connections(process_element, joblet_process, joblet.job_name)
    
    def _merge_components(self, main_process: ET.Element, joblet_process: ET.Element, joblet_name: str) -> None:
        """
        Merge components from joblet into main process
        
        Args:
            main_process: Main job process element
            joblet_process: Joblet process element  
            joblet_name: Name of joblet for unique naming
        """
        # Find all nodes in joblet
        for node in joblet_process.findall('.//node'):
            try:
                # Create a copy of the node
                node_copy = copy.deepcopy(node)
                
                # Update unique name to avoid conflicts
                unique_name = node_copy.get('uniqueName', '')
                if unique_name:
                    new_unique_name = f"{joblet_name}_{unique_name}"
                    node_copy.set('uniqueName', new_unique_name)
                
                # Update any internal references in the node
                self._update_node_references(node_copy, joblet_name)
                
                # Add to main process
                main_process.append(node_copy)
                
            except Exception as e:
                self.logger.warning(f"Error merging component from {joblet_name}: {e}")
    
    def _merge_connections(self, main_process: ET.Element, joblet_process: ET.Element, joblet_name: str) -> None:
        """
        Merge connections from joblet into main process
        
        Args:
            main_process: Main job process element
            joblet_process: Joblet process element
            joblet_name: Name of joblet for unique naming
        """
        # Find all connections in joblet
        for connection in joblet_process.findall('.//connection'):
            try:
                # Create a copy of the connection
                connection_copy = copy.deepcopy(connection)
                
                # Update connection references
                self._update_connection_references(connection_copy, joblet_name)
                
                # Add to main process
                main_process.append(connection_copy)
                
            except Exception as e:
                self.logger.warning(f"Error merging connection from {joblet_name}: {e}")
    
    def _update_node_references(self, node: ET.Element, joblet_prefix: str) -> None:
        """
        Update internal references within a node
        
        Args:
            node: Node element to update
            joblet_prefix: Prefix to add to references
        """
        try:
            # Update any elementParameter that references other components
            for param in node.findall('.//elementParameter'):
                param_name = param.get('name', '')
                param_value = param.get('value', '')
                
                # Check if this parameter might reference another component
                if any(ref_indicator in param_name.upper() for ref_indicator in ['TARGET', 'SOURCE', 'COMPONENT', 'UNIQUE_NAME']):
                    if param_value and not param_value.startswith('"'):
                        # Update the reference
                        new_value = f'"{joblet_prefix}_{param_value.strip("\"\'")}"'
                        param.set('value', new_value)
        except Exception as e:
            self.logger.debug(f"Error updating node references: {e}")
    
    def _update_connection_references(self, connection: ET.Element, joblet_prefix: str) -> None:
        """
        Update connection references with joblet prefix - improved version
        
        Args:
            connection: Connection element to update
            joblet_prefix: Prefix to add to references
        """
        try:
            # Update source and target references
            for attr in ['source', 'target']:
                old_ref = connection.get(attr)
                if old_ref:
                    new_ref = f"{joblet_prefix}_{old_ref}"
                    connection.set(attr, new_ref)
            
            # Update metaname if present
            metaname = connection.get('metaname')
            if metaname:
                new_metaname = f"{joblet_prefix}_{metaname}"
                connection.set('metaname', new_metaname)
                
        except Exception as e:
            self.logger.debug(f"Error updating connection references: {e}")
    
    def _save_consolidated_job(self, consolidated_root: ET.Element, output_file: Path) -> None:
        """
        Save consolidated job to file with proper formatting
        
        Args:
            consolidated_root: Consolidated XML root
            output_file: Output file path
        """
        try:
            # Create ElementTree and write to file
            tree = ET.ElementTree(consolidated_root)
            
            # Ensure proper XML formatting
            self._indent_xml(consolidated_root)
            
            # Write to file with XML declaration
            with open(output_file, 'wb') as f:
                tree.write(f, encoding='utf-8', xml_declaration=True)
                
            self.logger.info(f"Successfully saved consolidated job: {output_file}")
            
        except Exception as e:
            error_msg = f"Error saving consolidated job {output_file}: {e}"
            self.logger.error(error_msg)
            self.summary.errors.append(error_msg)
    
    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """
        Add indentation to XML for better readability
        
        Args:
            elem: XML element to indent
            level: Current indentation level
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report
        
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 70)
        report.append("TALEND JOB CONSOLIDATION SUMMARY REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Processing Time: {self.summary.processing_time:.2f} seconds")
        report.append("")
        
        # File Statistics
        report.append("FILE STATISTICS:")
        report.append(f"  Total files processed: {self.summary.total_files_processed}")
        report.append(f"  Main jobs found: {self.summary.main_jobs_found}")
        report.append(f"  Joblets found: {self.summary.joblets_found}")
        report.append(f"  Shared jobs found: {self.summary.shared_jobs_found}")
        report.append(f"  Consolidated jobs created: {self.summary.consolidated_jobs_created}")
        report.append("")
        
        # Job Details
        if self.main_jobs:
            report.append("MAIN JOBS FOUND:")
            for name, job in self.main_jobs.items():
                component_count = len(job.components)
                joblet_refs = len(job.joblet_references)
                report.append(f"  - {name}: {component_count} components, {joblet_refs} joblet references")
        
        if self.joblets:
            report.append("\nJOBLETS FOUND:")
            for name, job in self.joblets.items():
                component_count = len(job.components)
                report.append(f"  - {name}: {component_count} components")
        
        # Consolidation Results
        if self.summary.joblet_merges:
            report.append("\nCONSOLIDATION RESULTS:")
            for main_job, joblets in self.summary.joblet_merges.items():
                report.append(f"  {main_job}:")
                for joblet in joblets:
                    report.append(f"    ✓ Merged: {joblet}")
        
        # Input/Output Information
        report.append("\nINPUT FOLDERS:")
        for folder in self.input_folders:
            report.append(f"  - {folder}")
        report.append(f"\nOUTPUT FOLDER: {self.output_folder}")
        
        # Errors and Warnings
        if self.summary.errors:
            report.append("\nERRORS:")
            for i, error in enumerate(self.summary.errors, 1):
                report.append(f"  {i}. {error}")
        
        if self.summary.warnings:
            report.append("\nWARNINGS:")
            for i, warning in enumerate(self.summary.warnings, 1):
                report.append(f"  {i}. {warning}")
        
        if not self.summary.errors and not self.summary.warnings:
            report.append("\n✅ No errors or warnings encountered during processing.")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_summary_report(self) -> None:
        """
        Save summary report to file
        """
        try:
            report = self.generate_summary_report()
            
            # Save to output folder
            report_file = self.output_folder / "consolidation_summary.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Also log summary
            self.logger.info(f"Summary report saved to: {report_file}")
            self.logger.info("\n" + report)
            
        except Exception as e:
            error_msg = f"Error saving summary report: {e}"
            self.logger.error(error_msg)
            self.summary.errors.append(error_msg)
    
    def run(self) -> ConsolidationSummary:
        """
        Run the complete consolidation process
        
        Returns:
            Consolidation summary
        """
        try:
            self.logger.info("Starting Talend Job Consolidation process...")
            start_time = datetime.now()
            
            # Scan folders for job files
            self.scan_folders()
            
            # Consolidate jobs
            self.consolidate_jobs()
            
            # Calculate total processing time
            self.summary.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate and save summary
            self.save_summary_report()
            
            self.logger.info("Consolidation process completed successfully!")
            
        except Exception as e:
            error_msg = f"Fatal error during consolidation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.summary.errors.append(error_msg)
        
        return self.summary


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Talend Job Consolidator for Data Lineage')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input folders containing Talend job files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output folder for consolidated jobs')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create consolidator
    consolidator = TalendJobConsolidator(args.input, args.output)
    
    # Run consolidation
    summary = consolidator.run()
    
    # Print final summary
    print(f"\n🎯 CONSOLIDATION COMPLETED!")
    print(f"📁 Processed {summary.total_files_processed} files")
    print(f"🔗 Created {summary.consolidated_jobs_created} consolidated jobs")
    
    if summary.errors:
        print(f"❌ Encountered {len(summary.errors)} errors")
        return 1
    else:
        print("✅ No errors encountered")
        return 0


if __name__ == "__main__":
    exit(main())
