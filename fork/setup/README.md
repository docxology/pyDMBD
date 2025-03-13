# Setup Guide for DMBD Documentation Project

This guide will help you set up your development environment for working with the DMBD documentation project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv or conda (for virtual environment management)

## Setting Up Your Development Environment

### 1. Create and Activate a Virtual Environment

Using virtualenv:
```bash
# Create a new virtual environment
python -m venv dmbd-env

# Activate the virtual environment
# On Windows:
dmbd-env\Scripts\activate
# On Unix or MacOS:
source dmbd-env/bin/activate
```

Using conda:
```bash
# Create a new conda environment
conda create -n dmbd-env python=3.8

# Activate the conda environment
conda activate dmbd-env
```

### 2. Install Dependencies

Install all required packages:
```bash
# Install basic requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install testing dependencies
pip install -e ".[test]"

# Install documentation dependencies
pip install -e ".[docs]"
```

### 3. Verify Installation

Run the following commands to verify your installation:

```bash
# Check Python version
python --version

# Verify key packages
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
python -c "import pytest; print(f'Pytest version: {pytest.__version__}')"
```

### 4. Run Tests

To ensure everything is set up correctly, run the basic tests:

```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=docxology tests/
```

## Development Tools

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Run flake8 for style guide enforcement
flake8 .

# Run mypy for type checking
mypy .
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check your code before committing:

```bash
# Install pre-commit hooks
pre-commit install
```

## Building Documentation

To build the documentation locally:

```bash
# Navigate to the docs directory
cd docs

# Build the documentation
make html

# View the documentation (open _build/html/index.html in your browser)
```

## Troubleshooting

### Common Issues

1. **Package Import Errors**
   - Ensure your virtual environment is activated
   - Verify the package is installed in development mode
   - Check Python path settings

2. **CUDA/PyTorch Issues**
   - Verify CUDA installation if using GPU
   - Check PyTorch installation matches your CUDA version

3. **Documentation Build Errors**
   - Ensure all documentation dependencies are installed
   - Check for proper RST/MD syntax
   - Verify all referenced files exist

### Version Check Script

You can run the following script to check all package versions:

```python
# Save as version_check.py in the setup directory
import sys
import pkg_resources

def check_versions():
    """Print versions of all installed packages."""
    print(f"Python version: {sys.version}")
    print("\nInstalled packages:")
    for pkg in sorted(pkg_resources.working_set):
        print(f"{pkg.key} version: {pkg.version}")

if __name__ == "__main__":
    check_versions()
```

Run it with:
```bash
python version_check.py
```

## Getting Help

If you encounter any issues:

1. Check the [documentation](https://dmbd-docs.readthedocs.io/)
2. Search existing GitHub issues
3. Create a new issue with:
   - Your environment details
   - Steps to reproduce the problem
   - Error messages
   - Relevant logs 