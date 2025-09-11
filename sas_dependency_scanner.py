import re
import csv
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import pickle
import hashlib
import time
import os
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional
import gc

# ----------------------------
# Optimized Regex patterns (compiled once)
# ----------------------------
class RegexPatterns:
    def __init__(self):
        self.VAR_ASSIGN = re.compile(r'(?P<var>\w+)\s*=', re.I)
        self.LIBNAME = re.compile(r'\blibname\s+(?P<lib>\w+)\s+"(?P<path>[^"]+)"', re.I)
        self.INCLUDE = re.compile(r'%include\s+"?(?P<file>[^";]+)"?', re.I)
        self.DATA = re.compile(r'\bdata\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
        self.SET = re.compile(r'\bset\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
        self.PROC = re.compile(r'\bproc\s+(?P<proc>\w+)', re.I)
        self.PROC_SQL = re.compile(r'^\s*proc\s+sql', re.I)
        self.CREATE_TABLE = re.compile(r'\bcreate\s+table\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
        self.INSERT_INTO = re.compile(r'\binsert\s+into\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
        self.SELECT_INTO = re.compile(r'\bselect\s+.+into\s*:(?P<var>\w+)', re.I)
        self.LET = re.compile(r'%let\s+(?P<name>\w+)\s*=\s*(?P<val>[^;]+);?', re.I)
        self.MACRO_DEF = re.compile(r'%macro\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)?', re.I)
        self.MACRO_END = re.compile(r'%mend', re.I)
        self.MACRO_CALL = re.compile(r'%(?P<name>\w+)\s*(\((?P<args>[^)]*)\))?;?', re.I)
        self.SYMPUTX = re.compile(r'call\s+symputx?\s*\(\s*["\'](?P<var>\w+)["\']\s*,\s*(?P<val>[^)]+)\)', re.I)
        self.KEEP_DROP = re.compile(r'\b(?:keep|drop)\s*=\s*(?P<vars>[\w\s]+);', re.I)
        self.MACRO_VAR = re.compile(r'&+[\w]+')

# Global regex patterns instance
PATTERNS = RegexPatterns()

# ----------------------------
# File content cache with hash-based invalidation
# ----------------------------
class FileCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.max_memory_cache = 100  # Maximum files to keep in memory
        
    def _get_file_hash(self, fpath: Path) -> str:
        """Get file hash for cache invalidation"""
        try:
            stat = fpath.stat()
            return hashlib.md5(f"{fpath}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()
        except:
            return hashlib.md5(str(fpath).encode()).hexdigest()
    
    def get_content(self, fpath: Path) -> Optional[str]:
        """Get cached file content or None if not cached/invalid"""
        file_hash = self._get_file_hash(fpath)
        
        # Check memory cache first
        if str(fpath) in self.memory_cache:
            cached_hash, content = self.memory_cache[str(fpath)]
            if cached_hash == file_hash:
                return content
        
        # Check disk cache if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        content = pickle.load(f)
                        self._update_memory_cache(str(fpath), file_hash, content)
                        return content
                except:
                    try:
                        cache_file.unlink(missing_ok=True)
                    except:
                        pass
        
        return None
    
    def set_content(self, fpath: Path, content: str):
        """Cache file content"""
        file_hash = self._get_file_hash(fpath)
        self._update_memory_cache(str(fpath), file_hash, content)
        
        # Save to disk cache if available
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_dir / f"{file_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(content, f)
            except:
                pass  # Ignore cache write errors
    
    def _update_memory_cache(self, fpath_str: str, file_hash: str, content: str):
        """Update memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove oldest entry
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[fpath_str] = (file_hash, content)

# ----------------------------
# Optimized macro variable resolver
# ----------------------------
def iterative_macro_resolve(line: str, scope_dict: dict, max_passes: int = 10) -> Tuple[str, dict, set]:
    """Resolve macro variables with optimized performance"""
    used = {}
    unresolved = set()
    resolved = line
    
    # Quick check if line has any macro variables
    if '&' not in line:
        return resolved, used, unresolved
    
    for i in range(max_passes):
        changes = False
        matches = list(PATTERNS.MACRO_VAR.finditer(resolved))
        
        for m in matches:
            token = m.group(0)
            ampersands = len(re.match(r'&+', token).group(0))
            name = token[ampersands:]
            
            value = scope_dict.get(name.upper())
            if value is None:
                unresolved.add(token)
                continue
            
            used[token] = value
            if ampersands > 1:
                replacement = "&" * (ampersands - 1) + str(value)
            else:
                replacement = str(value)
            
            if replacement != token:
                resolved = resolved.replace(token, replacement, 1)
                changes = True
        
        if not changes:
            break
    
    return resolved, used, unresolved

# ----------------------------
# Parallel file processor
# ----------------------------
def process_single_file(args) -> Dict:
    """Process a single SAS file - designed for parallel execution"""
    fpath, collect_macros_only, cache_dir, global_macro_vars = args
    
    try:
        # Initialize local cache
        cache = FileCache(cache_dir) if cache_dir else None
        
        # Read file content (with caching)
        content = None
        if cache:
            content = cache.get_content(fpath)
        
        if content is None:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if cache:
                cache.set_content(fpath, content)
        
        lines = content.splitlines()
        
        result = {
            'file': str(fpath),
            'macro_vars': {},
            'macros': {},
            'producers': [],
            'consumers': [],
            'relationships': [],
            'unresolved': [],
            'replace_rules': {},
            'expanded_lines': [],
            'line_count': len(lines)
        }
        
        # Process based on mode
        if collect_macros_only:
            _collect_macros_from_content(lines, fpath, result)
        else:
            _parse_content_for_dependencies(lines, fpath, result, global_macro_vars or {})
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing {fpath}: {e}")
        return {'file': str(fpath), 'error': str(e), 'line_count': 0}

def _collect_macros_from_content(lines: List[str], fpath: Path, result: Dict):
    """Extract macros from file content"""
    in_macro = False
    macro_name = None
    
    for ln, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
            
        try:
            if not in_macro:
                if m := PATTERNS.LET.search(stripped):
                    name, val = m.group("name"), m.group("val").strip()
                    result['macro_vars'][name.upper()] = val
                elif m := PATTERNS.MACRO_DEF.search(stripped):
                    in_macro = True
                    macro_name = m.group("name").upper()
                    params_str = m.group("params") or ""
                    params = [p.strip() for p in params_str.split(",") if p.strip()]
                    
                    defaults, clean_params = {}, []
                    for p in params:
                        if "=" in p:
                            key, val = p.split("=", 1)
                            clean_params.append(key.strip().upper())
                            defaults[key.strip().upper()] = val.strip()
                        else:
                            clean_params.append(p.strip().upper())
                    
                    result['macros'][macro_name] = {
                        "params": clean_params, 
                        "defaults": defaults, 
                        "body": []
                    }
            else:
                if PATTERNS.MACRO_END.search(stripped):
                    in_macro = False
                    macro_name = None
                elif macro_name and macro_name in result['macros']:
                    result['macros'][macro_name]["body"].append(stripped)
        except Exception as e:
            logging.debug(f"Error collecting macros at line {ln} in {fpath}: {e}")

def _parse_content_for_dependencies(lines: List[str], fpath: Path, result: Dict, global_macro_vars: Dict):
    """Parse file content for dependencies"""
    in_sql = False
    expanded_lines = []
    
    # Combine global macro vars with any local ones
    local_scope = dict(global_macro_vars)
    local_scope.update(result.get('macro_vars', {}))
    
    for ln, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            expanded_lines.append(stripped)
            continue
        
        try:
            # Resolve macros in this line
            resolved_line, used_macros, unresolved_macros = iterative_macro_resolve(
                stripped, local_scope
            )
            expanded_lines.append(resolved_line)
            
            # Track macro replacements
            for token, replacement in used_macros.items():
                result['replace_rules'][token] = (replacement, str(fpath))
            
            # Track unresolved macros
            for token in unresolved_macros:
                result['unresolved'].append((token, str(fpath), ln, stripped))
            
            # Parse dependencies from resolved line
            line_lower = resolved_line.lower()
            
            # INCLUDE processing
            if 'include' in line_lower:
                if m := PATTERNS.INCLUDE.search(resolved_line):
                    tgt = m.group("file")
                    result['relationships'].append((str(fpath), tgt, "INCLUDE", str(fpath), ln, stripped))
            
            # LIBNAME processing  
            if 'libname' in line_lower:
                if m := PATTERNS.LIBNAME.search(resolved_line):
                    result['producers'].append((m.group("lib"), str(fpath), ln, stripped, "LIBNAME"))
            
            # DATA step processing
            if 'data ' in line_lower:
                if m := PATTERNS.DATA.search(resolved_line):
                    result['producers'].append((m.group("ds"), str(fpath), ln, stripped, "DATA"))
            
            # SET processing
            if 'set ' in line_lower:
                if m := PATTERNS.SET.search(resolved_line):
                    result['consumers'].append((m.group("ds"), str(fpath), ln, stripped, "SET"))
            
            # SQL processing
            if 'proc sql' in line_lower and PATTERNS.PROC_SQL.search(resolved_line):
                in_sql = True
            elif 'quit;' in line_lower:
                in_sql = False
            
            if in_sql:
                if 'create table' in line_lower:
                    if m := PATTERNS.CREATE_TABLE.search(resolved_line):
                        result['producers'].append((m.group("ds"), str(fpath), ln, stripped, "SQL_CREATE"))
                if 'insert into' in line_lower:
                    if m := PATTERNS.INSERT_INTO.search(resolved_line):
                        result['consumers'].append((m.group("ds"), str(fpath), ln, stripped, "SQL_INSERT"))
            
            # Macro calls
            if '%' in resolved_line:
                if m := PATTERNS.MACRO_CALL.search(resolved_line):
                    name = m.group("name").upper()
                    result['consumers'].append((name, str(fpath), ln, stripped, "MACRO_CALL"))
            
            # Variable assignments
            if '=' in resolved_line:
                if m := PATTERNS.VAR_ASSIGN.search(resolved_line):
                    result['producers'].append((m.group("var"), str(fpath), ln, stripped, "VAR"))
            
            # SYMPUTX
            if 'symput' in line_lower:
                if m := PATTERNS.SYMPUTX.search(resolved_line):
                    result['producers'].append((m.group("var"), str(fpath), ln, stripped, "SYMPUTX"))
            
            # SELECT INTO
            if 'into :' in line_lower:
                if m := PATTERNS.SELECT_INTO.search(resolved_line):
                    result['producers'].append((m.group("var"), str(fpath), ln, stripped, "SQL_INTO"))
            
            # KEEP/DROP
            if 'keep=' in line_lower or 'drop=' in line_lower:
                if m := PATTERNS.KEEP_DROP.search(resolved_line):
                    for v in m.group("vars").replace("\n", " ").split():
                        if v.strip():
                            result['consumers'].append((v.strip(), str(fpath), ln, stripped, "KEEP_DROP"))
                            
        except Exception as e:
            logging.debug(f"Error parsing line {ln} in {fpath}: {e}")
            expanded_lines.append(stripped)  # Add original line if processing fails
    
    result['expanded_lines'] = expanded_lines

# ----------------------------
# Enhanced SAS Dependency Scanner
# ----------------------------
class OptimizedSASDependencyScanner:
    def __init__(self, root_folder: str, output_folder: str, 
                 max_workers: Optional[int] = None,
                 enable_cache: bool = True,
                 file_patterns: List[str] = None,
                 exclude_patterns: List[str] = None):
        
        self.root_folder = Path(root_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Performance settings
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.enable_cache = enable_cache
        self.cache_dir = self.output_folder / ".cache" if enable_cache else None
        
        # File filtering
        self.file_patterns = file_patterns or ["*.sas"]
        self.exclude_patterns = exclude_patterns or []
        
        # Data storage - Initialize ALL attributes
        self.graph = nx.DiGraph()
        self.producers = []
        self.consumers = []
        self.relationships = []
        self.unresolved = []
        self.macro_vars = {}
        self.macros = {}
        self.replace_rules = {}
        self.expanded_files = {}  # FIXED: Initialize this attribute
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_lines': 0,
            'cache_hits': 0,
            'processing_time': 0
        }

    def _find_sas_files(self) -> List[Path]:
        """Find SAS files with pattern matching and exclusions"""
        all_files = []
        
        for pattern in self.file_patterns:
            all_files.extend(self.root_folder.rglob(pattern))
        
        # Apply exclusions
        if self.exclude_patterns:
            exclude_compiled = [re.compile(pattern) for pattern in self.exclude_patterns]
            filtered_files = []
            for f in all_files:
                exclude = False
                for exc_pattern in exclude_compiled:
                    if exc_pattern.search(str(f)):
                        exclude = True
                        break
                if not exclude:
                    filtered_files.append(f)
            all_files = filtered_files
        
        return sorted(set(all_files))  # Remove duplicates and sort

    def _merge_results(self, results: List[Dict]):
        """Merge results from parallel processing"""
        for result in results:
            if 'error' in result:
                logging.warning(f"Skipped file {result['file']} due to error: {result['error']}")
                continue
                
            # Merge macro variables and macros
            self.macro_vars.update(result.get('macro_vars', {}))
            self.macros.update(result.get('macros', {}))
            
            # Merge replace rules
            self.replace_rules.update(result.get('replace_rules', {}))
            
            # Store expanded files
            if result.get('expanded_lines'):
                self.expanded_files[result['file']] = result['expanded_lines']
            
            # Extend lists
            self.producers.extend(result.get('producers', []))
            self.consumers.extend(result.get('consumers', []))
            self.relationships.extend(result.get('relationships', []))
            self.unresolved.extend(result.get('unresolved', []))
            
            # Update stats
            self.stats['total_lines'] += result.get('line_count', 0)
            self.stats['files_processed'] += 1

    def _write_outputs_efficiently(self):
        """Write outputs with optimized I/O"""
        logging.info("Writing outputs...")
        
        # Write CSVs in parallel using ThreadPoolExecutor
        def write_csv_file(args):
            filename, header, rows = args
            filepath = self.output_folder / filename
            try:
                with open(filepath, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(header)
                    writer.writerows(rows)
                return filename
            except Exception as e:
                logging.error(f"Error writing {filename}: {e}")
                return None
        
        csv_tasks = [
            ("relationships.csv", ["source", "target", "relation", "file", "line_no", "context"], self.relationships),
            ("producers.csv", ["resource", "file", "line_no", "context", "kind"], self.producers),
            ("consumers.csv", ["resource", "file", "line_no", "context", "kind"], self.consumers),
            ("unresolved_macros.csv", ["token", "file", "line_no", "context"], self.unresolved)
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_csv_file, task) for task in csv_tasks]
            for future in as_completed(futures):
                try:
                    filename = future.result()
                    if filename:
                        logging.debug(f"Written {filename}")
                except Exception as e:
                    logging.error(f"Error writing CSV: {e}")

    def write_replace_csv(self):
        """Write macro replacement rules efficiently"""
        logging.info("Writing replace.csv")
        try:
            with open(self.output_folder / "replace.csv", "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerow(["pattern", "replacement", "scope"])
                for token, (repl, scope) in self.replace_rules.items():
                    writer.writerow([token, repl, scope])
            logging.debug(f"Written {len(self.replace_rules)} replacement rules")
        except Exception as e:
            logging.error(f"Error writing replace.csv: {e}")

    def write_expanded_files(self):
        """Write macro-expanded SAS files efficiently"""
        logging.info("Writing expanded SAS files")
        try:
            # Create expanded directory
            out_dir = self.output_folder / "expanded"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            all_lines = []
            files_written = 0
            
            # Use ThreadPoolExecutor for parallel file writing
            def write_expanded_file(args):
                fpath, lines = args
                try:
                    out_path = out_dir / Path(fpath).name
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(lines))
                    return len(lines)
                except Exception as e:
                    logging.error(f"Error writing expanded file {fpath}: {e}")
                    return 0
            
            # Prepare tasks for parallel execution
            file_tasks = list(self.expanded_files.items())
            
            if file_tasks:  # Only proceed if we have files to write
                with ThreadPoolExecutor(max_workers=min(8, len(file_tasks))) as executor:
                    futures = [executor.submit(write_expanded_file, task) for task in file_tasks]
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            line_count = future.result()
                            if i < len(file_tasks):
                                fpath, lines = file_tasks[i]
                                all_lines.extend([f"/* File: {Path(fpath).name} */"] + lines + [""])
                            files_written += 1
                        except Exception as e:
                            logging.error(f"Error writing expanded file: {e}")
                
                # Write combined file
                if all_lines:
                    with open(self.output_folder / "expanded_all.sas", "w", encoding="utf-8") as fh:
                        fh.write("\n".join(all_lines))
                
                logging.debug(f"Written {files_written} expanded files")
            else:
                logging.info("No expanded files to write")
                
        except Exception as e:
            logging.error(f"Error writing expanded files: {e}")

    def _create_smart_visualization(self):
        """Create visualization optimized for large graphs"""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        logging.info(f"Creating visualization for {num_nodes} nodes, {num_edges} edges")
        
        if num_nodes == 0:
            logging.warning("No nodes to visualize")
            return
        
        # Skip visualization for very large graphs to save time
        if num_nodes > 500:
            logging.warning(f"Graph too large ({num_nodes} nodes) - skipping visualization. Use GraphML for external tools.")
            return
        
        try:
            # Adaptive figure sizing
            fig_size = min(20, max(10, num_nodes * 0.3))
            plt.figure(figsize=(fig_size, fig_size))
            
            # Use faster layout algorithms for large graphs
            if num_nodes < 50:
                pos = nx.spring_layout(self.graph, k=2, iterations=100)
                node_size = 800
                font_size = 8
            elif num_nodes < 150:
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
                node_size = 400
                font_size = 6
            else:
                pos = nx.circular_layout(self.graph)  # Faster for large graphs
                node_size = 200
                font_size = 4
            
            # Draw graph
            nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, 
                                 node_color='lightblue', alpha=0.7)
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, 
                                 edge_color='gray', arrows=True)
            nx.draw_networkx_labels(self.graph, pos, font_size=font_size, 
                                  font_weight='bold')
            
            plt.title(f"SAS Dependencies ({num_nodes} nodes, {num_edges} edges)", 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.savefig(self.output_folder / "relationships.png", 
                       dpi=150,  # Reduced DPI for faster saving
                       bbox_inches='tight',
                       facecolor='white')
            
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
        finally:
            plt.close()

    def _write_summary_report(self):
        """Generate comprehensive summary report"""
        try:
            with open(self.output_folder / "summary_report.txt", "w", encoding="utf-8") as fh:
                fh.write("SAS Dependency Analysis Summary\n")
                fh.write("=" * 50 + "\n\n")
                fh.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.write(f"Processing time: {self.stats['processing_time']:.2f} seconds\n")
                fh.write(f"Files processed: {self.stats['files_processed']:,}\n")
                fh.write(f"Total lines analyzed: {self.stats['total_lines']:,}\n")
                fh.write(f"Cache hits: {self.stats['cache_hits']}\n\n")
                
                fh.write("Dependencies Found:\n")
                fh.write(f"  Producers: {len(self.producers):,}\n")
                fh.write(f"  Consumers: {len(self.consumers):,}\n")
                fh.write(f"  Relationships: {len(self.relationships):,}\n")
                fh.write(f"  Graph nodes: {self.graph.number_of_nodes():,}\n")
                fh.write(f"  Graph edges: {self.graph.number_of_edges():,}\n\n")
                
                fh.write(f"Macro Analysis:\n")
                fh.write(f"  Macro variables: {len(self.macro_vars)}\n")
                fh.write(f"  Macro definitions: {len(self.macros)}\n")
                fh.write(f"  Macro replacements: {len(self.replace_rules)}\n")
                fh.write(f"  Expanded files: {len(self.expanded_files)}\n")
                fh.write(f"  Unresolved references: {len(self.unresolved)}\n\n")
                
                # Top producers by type
                producer_types = defaultdict(int)
                for _, _, _, _, kind in self.producers:
                    producer_types[kind] += 1
                
                fh.write("Producer Types:\n")
                for ptype, count in sorted(producer_types.items(), key=lambda x: x[1], reverse=True):
                    fh.write(f"  {ptype}: {count}\n")
                
                fh.write("\n")
                
                # Top macro variables by usage
                if self.replace_rules:
                    macro_counts = defaultdict(int)
                    for token in self.replace_rules.keys():
                        macro_counts[token] += 1
                    
                    fh.write("Most Used Macro Variables:\n")
                    for token, count in sorted(macro_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]:
                        fh.write(f"  {token}: used in replacements\n")
                    fh.write("\n")
                
                # Top unresolved macros
                if self.unresolved:
                    unresolved_counts = defaultdict(int)
                    for token, _, _, _ in self.unresolved:
                        unresolved_counts[token] += 1
                    
                    fh.write("Top Unresolved Macros:\n")
                    for token, count in sorted(unresolved_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:15]:
                        fh.write(f"  {token}: {count} occurrences\n")
        except Exception as e:
            logging.error(f"Error writing summary report: {e}")

    def scan(self, quick_mode: bool = False):
        """Main scanning method with performance optimizations"""
        start_time = time.time()
        logging.info(f"Starting optimized SAS dependency scan in {self.root_folder}")
        
        # Find files
        sas_files = self._find_sas_files()
        logging.info(f"Found {len(sas_files)} SAS files")
        
        if not sas_files:
            logging.warning("No SAS files found")
            return
        
        try:
            # Phase 1: Collect macros (parallel)
            logging.info("Phase 1: Collecting macro definitions...")
            macro_args = [(f, True, self.cache_dir, {}) for f in sas_files]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                macro_results = list(executor.map(process_single_file, macro_args))
            
            self._merge_results(macro_results)
            logging.info(f"Collected {len(self.macro_vars)} macro variables and {len(self.macros)} macro definitions")
            
            # Phase 2: Parse dependencies (parallel)
            logging.info("Phase 2: Parsing dependencies...")
            parse_args = [(f, False, self.cache_dir, self.macro_vars) for f in sas_files]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                parse_results = list(executor.map(process_single_file, parse_args))
            
            self._merge_results(parse_results)
            
            # Build graph efficiently
            logging.info("Building dependency graph...")
            edges_to_add = [(s, t, {'relation': r, 'file': f, 'line_no': ln}) 
                           for s, t, r, f, ln, c in self.relationships]
            self.graph.add_edges_from(edges_to_add)
            
            # Generate outputs
            self._write_outputs_efficiently()
            self.write_replace_csv()
            
            if not quick_mode:
                # Write expanded files
                self.write_expanded_files()
                
                # Write GraphML
                try:
                    nx.write_graphml(self.graph, self.output_folder / "relationships.graphml")
                except Exception as e:
                    logging.error(f"Error writing GraphML: {e}")
                
                # Create visualization (if reasonable size)
                self._create_smart_visualization()
            
            # Always write summary
            self.stats['processing_time'] = time.time() - start_time
            self._write_summary_report()
            
            logging.info(f"Scan completed in {self.stats['processing_time']:.2f} seconds")
            logging.info(f"Processed {self.stats['files_processed']} files, {self.stats['total_lines']:,} lines")
            logging.info(f"Results saved to {self.output_folder}")
            
            # Additional statistics
            logging.info(f"Found {len(self.replace_rules)} macro replacements")
            if not quick_mode:
                logging.info(f"Generated {len(self.expanded_files)} expanded files")
                
        except Exception as e:
            logging.error(f"Error during scan: {e}")
            raise

# ----------------------------
# CLI with enhanced options
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Enterprise SAS Dependency Scanner")
    parser.add_argument("root", help="Root folder containing SAS files")
    parser.add_argument("--output", default="./output", help="Output folder")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--no-cache", action="store_true", help="Disable file caching")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip visualization)")
    parser.add_argument("--include", nargs="+", default=["*.sas"], 
                       help="File patterns to include")
    parser.add_argument("--exclude", nargs="+", 
                       help="File patterns to exclude (regex)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    try:
        scanner = OptimizedSASDependencyScanner(
            root_folder=args.root,
            output_folder=args.output,
            max_workers=args.workers,
            enable_cache=not args.no_cache,
            file_patterns=args.include,
            exclude_patterns=args.exclude
        )
        
        if args.profile:
            import cProfile
            cProfile.run('scanner.scan(quick_mode=args.quick)', 
                        filename=str(Path(args.output) / 'profile.stats'))
        else:
            scanner.scan(quick_mode=args.quick)
            
    except KeyboardInterrupt:
        logging.info("Scan interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise
