# Installation Guide

There are several ways to install and use the Dynamic Markov Blanket Detection (DMBD) package.

## Prerequisites

DMBD requires the following dependencies:
- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib (for visualization)

## Method 1: Clone the Repository (Recommended)

The recommended way to install DMBD is to clone the repository directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/bayesianempirimancer/pyDMBD.git

# Navigate to the repository directory
cd pyDMBD

# Install dependencies
pip install torch numpy matplotlib
```

## Method 2: Using as a Library

If you want to use DMBD in your own project without cloning the whole repository, you can:

```bash
# Create a directory for your project
mkdir my_dmbd_project
cd my_dmbd_project

# Download the core DMBD file
curl -O https://raw.githubusercontent.com/bayesianempirimancer/pyDMBD/main/DynamicMarkovBlanketDiscovery.py

# Create a simple script to use it
echo 'from DynamicMarkovBlanketDiscovery import DMBD' > example.py
```

## Verifying Installation

To verify that the installation is working correctly, you can run a simple test:

```bash
# From the pyDMBD directory
python -c "from DynamicMarkovBlanketDiscovery import DMBD; print('DMBD installed successfully!')"
```

## Common Issues

### ImportError: No module named 'dists'

DMBD relies on sub-modules in the repository. Make sure your working directory is the repository root or that the repository root is in your Python path.

### CUDA Issues

If you encounter CUDA-related errors, you might need to install a compatible version of PyTorch:

```bash
# For CUDA 11.6
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Development Installation

If you plan to contribute to DMBD, you might want to install it in development mode:

```bash
# Clone the repository
git clone https://github.com/bayesianempirimancer/pyDMBD.git
cd pyDMBD

# Install in development mode
pip install -e .
```

## Next Steps

Once you have successfully installed DMBD, check out the [Quick Start Guide](quick_start.md) to begin using it. 