"""Pytest configuration file with common fixtures and settings."""

import pytest
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so that the modules can be found
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Also add the fork directory itself for fork.examples import
fork_dir = parent_dir
if str(fork_dir) not in sys.path:
    sys.path.insert(0, str(fork_dir))

@pytest.fixture(scope="session")
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

@pytest.fixture(scope="session")
def sample_data():
    """Generate sample data for testing."""
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

@pytest.fixture(scope="session")
def torch_device():
    """Get the appropriate torch device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def torch_sample_data(torch_device):
    """Generate sample PyTorch tensors for testing."""
    n_samples = 100
    n_features = 5
    X = torch.randn(n_samples, n_features, device=torch_device)
    y = torch.randn(n_samples, device=torch_device)
    return X, y 