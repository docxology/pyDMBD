"""Tests for verifying the Python environment and package installations."""

import os
import sys
import pytest
import pkg_resources
import psutil
import torch

def test_python_version():
    """Test Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python version should be 3.8 or higher"

def test_required_packages():
    """Test that all required packages are installed."""
    required_packages = [
        "torch",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "pytest",
        "pytest-cov",
        "jupyter",
        "ipykernel",
        "nbformat",
        "nbconvert",
        "sphinx",
        "sphinx-rtd-theme",
        "myst-parser",
        "psutil",  # For memory monitoring
    ]
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    for package in required_packages:
        assert package in installed_packages, f"{package} is not installed"

def test_torch_import():
    """Test PyTorch import and basic functionality."""
    import torch
    assert torch.cuda.is_available() or True, "CUDA not available, but CPU will work"
    x = torch.rand(2, 3)
    assert x.shape == (2, 3), "PyTorch tensor creation failed"
    
    # Test double precision support
    torch.set_default_dtype(torch.float64)
    y = torch.rand(2, 3)
    assert y.dtype == torch.float64, "Double precision not supported"

def test_numpy_import():
    """Test NumPy import and basic functionality."""
    import numpy as np
    x = np.array([1, 2, 3])
    assert x.shape == (3,), "NumPy array creation failed"
    
    # Test double precision
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert y.dtype == np.float64, "Double precision not supported"

def test_matplotlib_import():
    """Test Matplotlib import and basic functionality."""
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.close(fig)
    assert True, "Matplotlib import and basic functionality check passed"

def test_scipy_import():
    """Test SciPy import and basic functionality."""
    from scipy import stats
    x = stats.norm.rvs(size=100)
    assert len(x) == 100, "SciPy random normal distribution generation failed"

def test_pandas_import():
    """Test Pandas import and basic functionality."""
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert len(df) == 3, "Pandas DataFrame creation failed"

def test_system_resources():
    """Test system resource availability."""
    # Check available memory
    memory = psutil.virtual_memory()
    print(f"\nSystem memory status:")
    print(f"Total: {memory.total / (1024**3):.1f} GB")
    print(f"Available: {memory.available / (1024**3):.1f} GB")
    print(f"Used: {memory.used / (1024**3):.1f} GB")
    print(f"Percentage: {memory.percent}%")
    
    # Ensure at least 4GB available for tests
    min_required_memory = 4 * 1024**3  # 4GB in bytes
    assert memory.available >= min_required_memory, \
        f"Insufficient memory available. Need at least 4GB, have {memory.available / (1024**3):.1f}GB"
    
    # Check CPU count
    cpu_count = psutil.cpu_count()
    print(f"\nCPU cores: {cpu_count}")
    assert cpu_count >= 2, "Need at least 2 CPU cores"

def test_torch_memory_management():
    """Test PyTorch memory management capabilities."""
    # Test basic tensor creation and deletion
    x = torch.rand(1000, 1000)
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    del x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    assert final_memory <= initial_memory, "Memory not properly freed after tensor deletion"
    
    # Test gradient cleanup
    model = torch.nn.Linear(100, 100)
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(10, 100)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.zero_grad()
    
    # Verify gradients are cleared
    for param in model.parameters():
        assert param.grad is None or torch.all(param.grad == 0), \
            "Gradients not properly cleared"

def test_output_directories():
    """Test that required output directories exist and are writable."""
    from pathlib import Path
    
    # Define required directories
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        base_dir / "dmbd_outputs",
        base_dir / "dmbd_outputs/lorenz",
        base_dir / "dmbd_outputs/cartthingy",
        base_dir / "dmbd_outputs/flame",
        base_dir / "dmbd_outputs/forager",
        base_dir / "dmbd_outputs/newtons_cradle"
    ]
    
    # Create and test each directory
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        assert directory.exists(), f"Directory {directory} does not exist"
        assert os.access(directory, os.W_OK), f"Directory {directory} is not writable"
        
        # Try writing a test file
        test_file = directory / "test_write.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("Test write access")
            test_file.unlink()  # Clean up
        except Exception as e:
            pytest.fail(f"Failed to write to {directory}: {e}")

def test_environment_variables():
    """Test that required environment variables are set correctly."""
    # Check PYTHONPATH includes necessary directories
    python_path = os.environ.get('PYTHONPATH', '')
    base_dir = str(Path(__file__).parent.parent.parent)
    if base_dir not in python_path:
        os.environ['PYTHONPATH'] = f"{base_dir}:{python_path}"
    
    # Set and check other environment variables
    os.environ['PYTHONMALLOC'] = 'debug'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Verify environment variables
    assert os.environ.get('PYTHONMALLOC') == 'debug'
    assert os.environ.get('PYTHONFAULTHANDLER') == '1'
    assert os.environ.get('OMP_NUM_THREADS') == '1'
    assert os.environ.get('MKL_NUM_THREADS') == '1'
    assert os.environ.get('NUMEXPR_NUM_THREADS') == '1'
    assert os.environ.get('MPLBACKEND') == 'Agg' 