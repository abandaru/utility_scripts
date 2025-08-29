# Installation Guide for Interactive Support Ticket Analyzer

## Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository, optional)

## Quick Start Installation

### Option 1: Minimal Installation (Recommended for first-time users)

```bash
# Install only essential packages
pip install pandas==2.1.4 numpy==1.24.3 plotly==5.17.0 openpyxl==3.1.2
```

### Option 2: Full Installation with Requirements File

```bash
# Clone or download the project files
# Then install all recommended packages
pip install -r requirements.txt
```

### Option 3: Minimal Installation with Requirements File

```bash
# Install only essential packages from file
pip install -r requirements-minimal.txt
```

## Detailed Installation Steps

### 1. Check Python Version

```bash
python --version
# or
python3 --version
```

Ensure you have Python 3.8 or higher.

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv ticket_analyzer_env

# Activate virtual environment
# On Windows:
ticket_analyzer_env\Scripts\activate
# On macOS/Linux:
source ticket_analyzer_env/bin/activate
```

### 3. Upgrade pip (Recommended)

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

Choose one of the installation methods above.

### 5. Verify Installation

```python
# Test imports
python -c "import pandas, numpy, plotly, openpyxl; print('All packages installed successfully!')"
```

## Package Descriptions

| Package | Purpose | Required |
|---------|---------|----------|
| **pandas** | Data manipulation and analysis | ✅ Yes |
| **numpy** | Numerical computing | ✅ Yes |
| **plotly** | Interactive visualizations | ✅ Yes |
| **openpyxl** | Excel file reading/writing | ✅ Yes |
| **kaleido** | Static image export for Plotly | 🔶 Optional |
| **python-dateutil** | Date/time utilities | 🔶 Optional |

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'openpyxl'"
**Solution:**
```bash
pip install openpyxl
```

#### Issue: Plotly charts not displaying in Jupyter
**Solution:**
```bash
pip install jupyterlab "ipywidgets>=7,<8"
```

#### Issue: Memory errors with large Excel files
**Solution:**
```bash
pip install pyarrow  # For better memory management
```

#### Issue: Slow performance with large datasets
**Solution:**
```bash
pip install numba  # For faster numerical computations
```

### Platform-Specific Notes

#### Windows Users
- Use `python` instead of `python3`
- Install Microsoft Visual C++ Build Tools if you encounter compilation errors
- Consider using Anaconda/Miniconda for easier package management

#### macOS Users
- You might need to install Xcode Command Line Tools: `xcode-select --install`
- Use `python3` and `pip3` explicitly

#### Linux Users
- Install development headers: `sudo apt-get install python3-dev` (Ubuntu/Debian)
- For CentOS/RHEL: `sudo yum install python3-devel`

## Alternative Installation Methods

### Using Conda (Anaconda/Miniconda)

```bash
# Create conda environment
conda create -n ticket_analyzer python=3.9

# Activate environment
conda activate ticket_analyzer

# Install packages
conda install pandas numpy plotly openpyxl -c conda-forge
```

### Using Poetry

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.8"
pandas = "^2.0.0"
numpy = "^1.21.0"
plotly = "^5.17.0"
openpyxl = "^3.0.9"
```

```bash
poetry install
```

## Development Installation

For contributors or advanced users:

```bash
# Install with development tools
pip install -r requirements.txt
pip install pytest black flake8 jupyter notebook

# Install in development mode
pip install -e .
```

## Performance Optimization

For better performance with large datasets:

```bash
# Install performance packages
pip install numba pyarrow fastparquet
```

## Verification Script

Create a file `test_installation.py`:

```python
#!/usr/bin/env python3
"""Test script to verify installation"""

def test_imports():
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import openpyxl
        print("✅ All required packages imported successfully!")
        
        # Test basic functionality
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        fig = go.Figure()
        print("✅ Basic functionality test passed!")
        
        # Version info
        print(f"\nPackage Versions:")
        print(f"pandas: {pd.__version__}")
        print(f"numpy: {np.__version__}")
        print(f"plotly: {go.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
```

Run the test:
```bash
python test_installation.py
```

## Getting Help

If you encounter issues:

1. **Check Python version**: Ensure you're using Python 3.8+
2. **Update pip**: `pip install --upgrade pip`
3. **Clear pip cache**: `pip cache purge`
4. **Use virtual environment**: Avoid conflicts with system packages
5. **Check error messages**: Read the full error output
6. **Search for solutions**: Google the specific error message

## Next Steps

After successful installation:

1. Download or create your Excel file with support ticket data
2. Run the main analysis script: `python interactive_ticket_analyzer.py`
3. Open the generated HTML files in your web browser
4. Explore the interactive dashboards!

## System Requirements

- **RAM**: 4GB minimum (8GB+ recommended for large datasets)
- **Storage**: 500MB free space for packages and output files
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge) for viewing interactive charts
- **Excel**: Microsoft Excel or LibreOffice Calc (for creating input files)
