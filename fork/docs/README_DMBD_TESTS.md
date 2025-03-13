# Dynamic Markov Blanket Detection (DMBD) Test Suite

This directory contains an enhanced test suite for the Dynamic Markov Blanket Detection algorithm, focusing on inference and visualization capabilities.

## Overview

The Dynamic Markov Blanket Detection (DMBD) algorithm identifies causal structures in multivariate time series data by discovering "Markov blankets" - boundaries that separate a system from its environment. This test suite provides comprehensive testing and visualization of DMBD's capabilities across various simulation examples.

## Key Features

- **Enhanced Visualizations**: Detailed visualizations of DMBD results including:
  - Markov blanket structure diagrams
  - Role assignments (sensor/boundary/internal) over time
  - Reconstruction quality comparisons
  - Principal component analysis of inferred states
  - Time-series animations showing changing assignments

- **Memory Management**: Improved memory handling to prevent test failures
  - Resource limiting to prevent out-of-memory errors
  - Thread control to reduce memory consumption
  - Timeout management for long-running tests

- **Model Evaluation**: Automatic assessment of model quality through:
  - ELBO (Evidence Lower Bound) tracking and visualization
  - Comparison of multiple model configurations
  - Reconstruction accuracy metrics

## Running the Tests

### Basic Usage

To run all DMBD tests with default settings:

```bash
python run_dmbd_tests.py
```

### Advanced Options

```bash
python run_dmbd_tests.py --timeout 1800 --memory-limit 12288 --seed 42
```

Options:
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Directory to store outputs (default: "dmbd_outputs")
- `--timeout`: Timeout in seconds per test (default: 1200)
- `--memory-limit`: Memory limit in MB per test (default: 8192)
- `--test-pattern`: Pattern to match test files (default: "test_*dmbd*")
- `--visualize-only`: Only run tests with visualization and skip analysis
- `--quick`: Run with reduced iterations and faster settings

## Example Suite

The test suite includes several example systems:

1. **Lorenz Attractor**: A chaotic system with three variables showing complex dynamics
2. **Newton's Cradle**: A physical simulation of a pendulum system with complex interactions
3. **Flame Example**: A simulation of particles influenced by a flame-like central force
4. **Forager**: An agent-based model of foraging behavior

## Visualization Outputs

For each system, the test suite generates the following visualizations:

- **Markov Blanket Structure**: Depicting causal links between variables
- **Role Assignments**: Showing which variables are classified as:
  - Sensor (s): Variables sensing the environment
  - Boundary (b): Variables forming the Markov blanket
  - Internal (z): Variables inside the system boundary

- **Reconstruction Plots**: Comparing original data with DMBD reconstruction
- **PC Scores**: Principal component analysis of inferred states
- **ELBO Convergence**: Tracking model training progress

## Understanding the Results

The DMBD algorithm partitions variables into three key roles:

1. **Sensor (s)**: Variables that sense external information but don't influence internal states
2. **Boundary (b)**: Variables that form the causal boundary between internal and external states
3. **Internal (z)**: Variables inside the boundary that don't directly interact with external states

The visualizations help interpret:
- How the algorithm has partitioned the system
- Whether the discovered Markov blanket aligns with the known physical structure
- How role assignments change over time
- How well the model reconstructs the original data

## Technical Notes

### Memory Management

The tests use several techniques to manage memory:
- Setting `PYTHONMALLOC=debug` to catch memory issues
- Limiting OpenMP and MKL threads to reduce parallel memory usage
- Setting resource limits to prevent OOM errors

### Customizing Tests

To modify the DMBD analysis parameters:
1. Edit the `run_dmbd_analysis` function in `tests/test_examples.py`
2. Adjust the `role_dims` and `hidden_dims` parameters to control model capacity
3. Change `training_iters` to control how long the model trains

## Requirements

This test suite requires:
- Python 3.7+
- PyTorch 1.8+
- Matplotlib
- NumPy
- pytest and related plugins (pytest-timeout, pytest-randomly, etc.)

## Licensing

This test suite is released under the same license as the main repository. 