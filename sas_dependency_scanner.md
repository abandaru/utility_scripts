
# SAS Dependency Scanner - Functionality Summary

## Primary Purpose

This tool analyzes SAS codebases to extract, visualize, and document dependencies between files, datasets, variables, and macros. It's designed for enterprise-level SAS code analysis and documentation.

## Core Functionality

### 1. **Code Parsing & Analysis**

- **File Discovery**: Recursively scans directories for `.sas` files
- **Multi-pass Analysis**: First collects macro definitions, then parses dependencies
- **Macro Resolution**: Resolves complex macro variables (`&var`, `&&var`, etc.) with iterative substitution
- **SQL Detection**: Identifies and properly handles PROC SQL blocks

### 2. **Dependency Extraction**

**Producers** (things that create resources):

- `DATA` steps creating datasets
- `LIBNAME` statements defining libraries
- Variable assignments
- `CREATE TABLE` statements
- Macro variable definitions (`%LET`, `SYMPUTX`)

**Consumers** (things that use resources):

- `SET` statements reading datasets
- `INSERT INTO` operations
- Variable references
- Macro calls
- `KEEP`/`DROP` variable lists

**Relationships**:

- File includes (`%INCLUDE`)
- Inter-file dependencies
- Macro call hierarchies

### 3. **Macro Processing**

- **Macro Definition Collection**: Captures `%MACRO` definitions with parameters and defaults
- **Macro Call Simulation**: Expands macro calls with parameter substitution
- **Variable Tracking**: Monitors `%LET` statements and their scope
- **Unresolved Reference Tracking**: Documents macro variables that couldn't be resolved

### 4. **Output Generation**

**CSV Reports**:

- `producers.csv` - What creates each resource
- `consumers.csv` - What uses each resource
- `relationships.csv` - Direct file dependencies
- `replace.csv` - Macro variable substitutions performed
- `unresolved_macros.csv` - Macro references that couldn't be resolved

**Visualizations**:

- `relationships.png` - Network graph of dependencies
- `relationships.graphml` - Graph data for external visualization tools

**Expanded Code**:

- `expanded/` directory with macro-resolved SAS files
- `expanded_all.sas` - All code combined with macros expanded

**Summary**:

- `summary_report.txt` - Analysis statistics and top unresolved macros

## Key Features

### **Enterprise-Ready**

- Handles large codebases with thousands of files
- Robust error handling and logging
- UTF-8 encoding support for international characters

### **Intelligent Parsing**

- Context-aware SQL vs. DATA step detection
- Multi-level macro variable resolution
- Recursive file include processing
- Parameter handling for macro calls

### **Comprehensive Documentation**

- Tracks line numbers and context for all findings
- Maps variable scopes and replacement rules
- Identifies circular dependencies and missing references

## Use Cases

1. **Code Documentation**: Generate comprehensive dependency maps for legacy SAS systems
2. **Impact Analysis**: Understand what will be affected by changes to specific datasets or macros
3. **Code Migration**: Identify all dependencies when moving or refactoring SAS code
4. **Quality Assurance**: Find unresolved macro references and broken dependencies
5. **Architecture Understanding**: Visualize the structure and flow of complex SAS applications

## Usage

bash

```bash
# Basic analysis
python sas_dependency_scanner.py /path/to/sas/code

# With detailed logging and custom output
python sas_dependency_scanner.py /path/to/sas/code --output ./analysis --debug
```

This tool essentially creates a comprehensive "blueprint" of SAS codebases, making complex enterprise SAS systems more understandable and maintainable.
