import re
import csv
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Regex patterns for SAS parsing
# ----------------------------
RE_VAR_ASSIGN = re.compile(r'(?P<var>\w+)\s*=', re.I)
RE_LIBNAME = re.compile(r'\blibname\s+(?P<lib>\w+)\s+"(?P<path>[^"]+)"', re.I)
RE_INCLUDE = re.compile(r'%include\s+"?(?P<file>[^";]+)"?', re.I)
RE_DATA = re.compile(r'\bdata\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
RE_SET = re.compile(r'\bset\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
RE_PROC = re.compile(r'\bproc\s+(?P<proc>\w+)', re.I)
RE_PROC_SQL = re.compile(r'^\s*proc\s+sql', re.I)
RE_CREATE_TABLE = re.compile(r'\bcreate\s+table\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
RE_INSERT_INTO = re.compile(r'\binsert\s+into\s+(?P<ds>\w+(?:\.\w+)*)', re.I)
RE_SELECT_INTO = re.compile(r'\bselect\s+.+into\s*:(?P<var>\w+)', re.I)
RE_LET = re.compile(r'%let\s+(?P<name>\w+)\s*=\s*(?P<val>[^;]+);?', re.I)
RE_MACRO_DEF = re.compile(r'%macro\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)?', re.I)
RE_MACRO_END = re.compile(r'%mend', re.I)
RE_MACRO_CALL = re.compile(r'%(?P<name>\w+)\s*(\((?P<args>[^)]*)\))?;?', re.I)
RE_SYMPUTX = re.compile(
    r'call\s+symputx?\s*\(\s*["\'](?P<var>\w+)["\']\s*,\s*(?P<val>[^)]+)\)',
    re.I
)
RE_KEEP_DROP = re.compile(r'\b(?:keep|drop)\s*=\s*(?P<vars>[\w\s]+);', re.I)

# ----------------------------
# Utility: Macro variable resolver (fixed + logged)
# ----------------------------
def iterative_macro_resolve(line, scope, max_passes=50):
    """Resolve macro variables (&var, &&var, &&&&var...) with multi-level resolution."""
    used = {}
    unresolved = set()
    resolved = line

    for i in range(max_passes):
        changes = False
        matches = list(re.finditer(r'&+[\w]+', resolved))
        for m in matches:
            token = m.group(0)
            ampersands = len(re.match(r'&+', token).group(0))
            name = token[ampersands:]

            value = scope.get(name.upper())
            if value is None:
                unresolved.add(token)
                continue

            used[token] = value
            if ampersands > 1:
                replacement = "&" * (ampersands - 1) + value
            else:
                replacement = value
            if replacement != token:
                resolved = resolved.replace(token, replacement, 1)
                changes = True

        if not changes:
            break

    if i == max_passes - 1:
        logging.warning("Max macro resolution passes reached for line: %s", line)

    if unresolved:
        logging.warning("Unresolved macros in line: '%s' -> %s", line, sorted(unresolved))

    if used:
        logging.debug("Resolved line: '%s' -> '%s' with %s", line, resolved, used)

    return resolved, used, unresolved


# ----------------------------
# SAS Dependency Scanner
# ----------------------------
class SASDependencyScanner:
    def __init__(self, root_folder, output_folder):
        self.root_folder = Path(root_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()

        self.producers = []
        self.consumers = []
        self.relationships = []
        self.unresolved = []  # collect unresolved macro tokens

        self.macro_vars = {}
        self.macros = {}
        self.replace_rules = {}
        self.expanded_files = {}

    # ----------------------------
    # Macro collection pass
    # ----------------------------
    def _collect_macros(self, fpath: Path):
        logging.info("Collecting macros from %s", fpath)
        in_macro, macro_name, macro_params, body = False, None, [], []
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not in_macro:
                    if m := RE_LET.search(line):
                        name, val = m.group("name"), m.group("val").strip()
                        self.macro_vars[name.upper()] = val
                        self.replace_rules[f"&{name.upper()}"] = (val, str(fpath))
                        logging.debug("Collected %%LET %s=%s", name, val)
                    elif m := RE_MACRO_DEF.search(line):
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
                        self.macros[macro_name] = {"params": clean_params, "defaults": defaults, "body": []}
                        logging.debug("Defined macro %s with params %s", macro_name, clean_params)
                else:
                    if RE_MACRO_END.search(line):
                        logging.debug("End of macro %s", macro_name)
                        in_macro, macro_name, macro_params, body = False, None, [], []
                    else:
                        if macro_name in self.macros:
                            self.macros[macro_name]["body"].append(line.strip())

    # ----------------------------
    # Macro call simulation (FIXED: handle 3-tuple return)
    # ----------------------------
    def _simulate_macro_call(self, name, args, caller_file):
        logging.info("Simulating macro call %s(%s)", name, args)
        if name not in self.macros:
            logging.warning("Macro %s not defined", name)
            return
            
        macro = self.macros[name]
        param_map = {}
        for idx, param in enumerate(macro["params"]):
            if idx < len(args):
                param_map[param.upper()] = args[idx]
            elif param.upper() in macro["defaults"]:
                param_map[param.upper()] = macro["defaults"][param.upper()]
        scope = dict(self.macro_vars)
        scope.update(param_map)

        expanded_body = []
        for bline in macro["body"]:
            resolved, used, _ = iterative_macro_resolve(bline, scope)  # FIXED: unpack 3 values
            expanded_body.append(resolved)
            for token, repl in used.items():
                self.replace_rules[token] = (repl, f"macro::{name}")
        self.expanded_files.setdefault(str(caller_file), []).extend(expanded_body)

    # ----------------------------
    # File parsing (FIXED: correct regex group names)
    # ----------------------------
    def parse_file(self, fpath: Path):
        logging.info("Parsing file %s", fpath)
        in_sql, expanded_lines = False, []
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for ln, line in enumerate(fh, start=1):
                stripped = line.strip()
                resolved_line, used, unresolved = iterative_macro_resolve(stripped, self.macro_vars)
                for token, repl in used.items():
                    self.replace_rules[token] = (repl, str(fpath))
                for token in unresolved:
                    self.unresolved.append((token, str(fpath), ln, stripped))
                
                try:
                    if m := RE_INCLUDE.search(resolved_line):
                        tgt = m.group("file")
                        self.relationships.append((str(fpath), tgt, "INCLUDE", str(fpath), ln, stripped))
                        include_path = (fpath.parent / tgt).resolve()
                        if include_path.exists():
                            self.parse_file(include_path)  # recursive include
                        else:
                            logging.warning("Include file not found: %s (referenced in %s)", tgt, fpath)
                    if m := RE_LIBNAME.search(resolved_line):
                        self.producers.append((m.group("lib"), str(fpath), ln, stripped, "LIBNAME"))
                    if m := RE_DATA.search(resolved_line):
                        self.producers.append((m.group("ds"), str(fpath), ln, stripped, "DATA"))  # FIXED: ds not name
                    if m := RE_SET.search(resolved_line):
                        self.consumers.append((m.group("ds"), str(fpath), ln, stripped, "SET"))
                    if RE_PROC_SQL.search(resolved_line):
                        in_sql = True
                    if in_sql and (m := RE_CREATE_TABLE.search(resolved_line)):
                        self.producers.append((m.group("ds"), str(fpath), ln, stripped, "SQL_CREATE"))  # FIXED: ds not name
                    if in_sql and (m := RE_INSERT_INTO.search(resolved_line)):
                        self.consumers.append((m.group("ds"), str(fpath), ln, stripped, "SQL_INSERT"))  # FIXED: ds not name
                    if "quit;" in resolved_line.lower():
                        in_sql = False
                    if m := RE_MACRO_CALL.search(resolved_line):
                        name = m.group("name").upper()
                        args = []
                        if m.group("args"):
                            args = [a.strip() for a in m.group("args").split(",")]
                        if name in self.macros:
                            self._simulate_macro_call(name, args, fpath)
                        else:
                            # Track macro calls even if not defined (external macros)
                            self.consumers.append((name, str(fpath), ln, stripped, "MACRO_CALL"))
                    if m := RE_SYMPUTX.search(resolved_line):
                        self.producers.append((m.group("var"), str(fpath), ln, stripped, "SYMPUTX"))
                    if m := RE_SELECT_INTO.search(resolved_line):
                        self.producers.append((m.group("var"), str(fpath), ln, stripped, "SQL_INTO"))
                    if m := RE_KEEP_DROP.search(resolved_line):
                        for v in m.group("vars").replace("\n", " ").split():
                            if v.strip():  # FIXED: check for empty strings
                                self.consumers.append((v.strip(), str(fpath), ln, stripped, "KEEP_DROP"))
                    if m := RE_VAR_ASSIGN.search(resolved_line):
                        self.producers.append((m.group("var"), str(fpath), ln, stripped, "VAR"))
                except Exception as e:
                    logging.error("Error parsing line %s in %s: %s", ln, fpath, e)
                expanded_lines.append(resolved_line)
        self.expanded_files[str(fpath)] = expanded_lines

    # ----------------------------
    # Outputs
    # ----------------------------
    def write_csv(self):
        logging.info("Writing CSV outputs")
        def write(name, header, rows):
            with open(self.output_folder / name, "w", newline="", encoding="utf-8") as fh:  # FIXED: add encoding
                csv.writer(fh).writerows([header] + rows)
        write("relationships.csv", ["source","target","relation","file","line_no","context"], self.relationships)
        write("producers.csv", ["resource","file","line_no","context","kind"], self.producers)
        write("consumers.csv", ["resource","file","line_no","context","kind"], self.consumers)

    def write_replace_csv(self):
        logging.info("Writing replace.csv")
        with open(self.output_folder / "replace.csv", "w", newline="", encoding="utf-8") as fh:  # FIXED: add encoding
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(["pattern","replacement","scope"])
            for token, (repl, scope) in self.replace_rules.items():
                writer.writerow([token, repl, scope])

    def write_expanded_files(self):
        logging.info("Writing expanded SAS files")
        out_dir = self.output_folder / "expanded"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_lines = []
        for fpath, lines in self.expanded_files.items():
            out_path = out_dir / Path(fpath).name
            with open(out_path, "w", encoding="utf-8") as fh:  # FIXED: add encoding
                fh.write("\n".join(lines))
            all_lines.extend(lines)
        with open(self.output_folder / "expanded_all.sas", "w", encoding="utf-8") as fh:  # FIXED: add encoding
            fh.write("\n".join(all_lines))

    def write_graphml(self):
        logging.info("Writing GraphML")
        for s,t,r,f,ln,c in self.relationships:
            self.graph.add_edge(s, t, relation=r, file=f, line_no=ln)
        nx.write_graphml(self.graph, self.output_folder / "relationships.graphml")

    def visualize_graph(self):
        logging.info("Saving graph visualization")
        if self.graph.number_of_nodes() == 0:
            logging.warning("No graph nodes to visualize")
            return
            
        try:
            # Create figure with appropriate size based on number of nodes
            num_nodes = self.graph.number_of_nodes()
            fig_size = min(16, max(8, num_nodes * 0.5))
            
            plt.figure(figsize=(fig_size, fig_size))
            
            # Choose layout based on graph size
            if num_nodes < 20:
                pos = nx.spring_layout(self.graph, k=2, iterations=50)
                node_size = 1000
                font_size = 10
            elif num_nodes < 50:
                pos = nx.spring_layout(self.graph, k=1.5, iterations=30)
                node_size = 600
                font_size = 8
            else:
                pos = nx.spring_layout(self.graph, k=1, iterations=20)
                node_size = 300
                font_size = 6
            
            # Draw the graph
            nx.draw(self.graph, pos, 
                   with_labels=True, 
                   node_size=node_size, 
                   font_size=font_size, 
                   arrows=True,
                   node_color='lightblue',
                   edge_color='gray',
                   font_weight='bold')
            
            plt.title("SAS Dependencies Graph", fontsize=14, fontweight='bold')
            
            # Save with tight bounding box (no need for tight_layout with networkx)
            plt.savefig(self.output_folder / "relationships.png", 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            
            logging.info("Graph visualization saved with %d nodes and %d edges", 
                        num_nodes, self.graph.number_of_edges())
            
        except Exception as e:
            logging.error("Error creating graph visualization: %s", e)
        finally:
            plt.close()  # Always close the figure to free memory

    def write_unresolved_csv(self):
        logging.info("Writing unresolved_macros.csv")
        with open(self.output_folder / "unresolved_macros.csv", "w", newline="", encoding="utf-8") as fh:  # FIXED: add encoding
            writer = csv.writer(fh)
            writer.writerow(["token", "file", "line_no", "context"])
            writer.writerows(self.unresolved)

    def write_summary_report(self):
        """ADDED: Generate a summary report of the analysis"""
        logging.info("Writing summary report")
        with open(self.output_folder / "summary_report.txt", "w", encoding="utf-8") as fh:
            fh.write("SAS Dependency Analysis Summary\n")
            fh.write("=" * 40 + "\n\n")
            fh.write(f"Files analyzed: {len(self.expanded_files)}\n")
            fh.write(f"Producers found: {len(self.producers)}\n")
            fh.write(f"Consumers found: {len(self.consumers)}\n")
            fh.write(f"Relationships found: {len(self.relationships)}\n")
            fh.write(f"Macro variables defined: {len(self.macro_vars)}\n")
            fh.write(f"Macros defined: {len(self.macros)}\n")
            fh.write(f"Unresolved macro references: {len(self.unresolved)}\n\n")
            
            if self.unresolved:
                fh.write("Top unresolved macros:\n")
                unresolved_counts = {}
                for token, _, _, _ in self.unresolved:
                    unresolved_counts[token] = unresolved_counts.get(token, 0) + 1
                for token, count in sorted(unresolved_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    fh.write(f"  {token}: {count} occurrences\n")

    # ----------------------------
    # Main driver
    # ----------------------------
    def scan(self):
        logging.info("Starting SAS dependency scan in %s", self.root_folder)
        
        # Collect macros first
        sas_files = list(self.root_folder.rglob("*.sas"))
        logging.info("Found %d SAS files", len(sas_files))
        
        for sas_file in sas_files:
            if sas_file.exists():
                self._collect_macros(sas_file)
        
        # Parse files
        for sas_file in sas_files:
            if sas_file.exists():
                self.parse_file(sas_file)
        
        # Generate outputs
        self.write_csv()
        self.write_replace_csv()
        # self.write_expanded_files()  # uncomment if required
        self.write_graphml()
        self.write_unresolved_csv()
        self.write_summary_report()  
        self.visualize_graph()
        
        logging.info("Scan complete. Outputs in %s", self.output_folder)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise SAS Dependency Scanner")
    parser.add_argument("root", help="Root folder containing SAS files")
    parser.add_argument("--output", default="./output", help="Output folder")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    try:
        SASDependencyScanner(args.root, args.output).scan()
    except Exception as e:
        logging.error("Fatal error during scan: %s", e)
        raise
