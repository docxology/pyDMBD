# Gaussian Blob Example for DMBD

This directory contains an example application of the Dynamic Markov Blanket Detection (DMBD) algorithm to a simulated moving Gaussian blob. The example demonstrates how DMBD can discover the causal structure in a simple dynamic system with a moving blob of activation.

## Overview

The Gaussian Blob simulation:

1. Creates a 2D grid environment
2. Animates a Gaussian blob moving along a circular path
3. Adds random noise to the observations
4. Labels grid cells as internal (blob), blanket (boundary), or external

The DMBD algorithm then:

1. Analyzes the time series data
2. Discovers the underlying Markov blanket structure
3. Assigns roles to each part of the system
4. Produces visualizations comparing its discoveries with ground truth

## Running the Example

### Basic Usage

To run the Gaussian Blob example with default settings:

```bash
cd fork
python run_gaussian_blob.py
```

This will:

- Create a simulation with default parameters
- Run the DMBD algorithm on the generated data
- Produce visualizations and save results to the output directory

### Command Line Options

The script supports several options for customization:

```bash
python run_gaussian_blob.py [--output-dir DIR] [--grid-size N] 
                            [--time-steps N] [--seed N]
                            [--sigma VALUE] [--noise-level VALUE]
                            [--convergence-attempts N] [--save-interval N]
                            [--verbose]
```

Options:

- `--output-dir`: Directory for saving outputs (default: `outputs/gaussian_blob`)
- `--grid-size`: Size of the grid (default: 16, higher values create more detailed simulations)
- `--time-steps`: Number of simulation steps (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--sigma`: Width of the Gaussian blob (default: 2.0)
- `--noise-level`: Level of noise added to observations (default: 0.02)
- `--convergence-attempts`: Number of configurations to try for DMBD convergence (default: 3)
- `--save-interval`: Interval for saving visualizations (default: 10)
- `--verbose`: Enable detailed output during training

### Quick Test

For a quick test with minimal settings:

```python
# In Python
from run_gaussian_blob import run_test
results = run_test(grid_size=8, time_steps=20)
print(f"Test succeeded: {results['success']}")
```

## Output Structure

Results are organized in the specified output directory:

```text
outputs/gaussian_blob/
├── raw_data/                  # Raw simulation data
│   ├── blob_animation.gif     # Animation of the raw simulation
│   ├── blob_frame_*.png       # Individual frames from the simulation
│   └── ...
├── sample_frames/             # Sample visualization frames
│   ├── ...
├── dmbd_results/              # DMBD analysis results
│   ├── comparisons/           # Detailed comparison visualizations
│   │   ├── ...
│   ├── raw_assignments/       # Raw DMBD role assignments
│   │   ├── ...
│   ├── dmbd_comparison.gif    # Animation comparing ground truth vs. DMBD
│   ├── dmbd_comparison_t*.png # Individual comparison frames
│   └── dmbd_results.pt        # Saved PyTorch model and results
└── run_log.txt                # Detailed log of the run
```

## Visualizations

The example generates several key visualizations:

1. **Raw Simulation Data**: Animation showing the moving Gaussian blob
2. **DMBD Comparisons**: Side-by-side comparisons of ground truth vs. DMBD-discovered roles
3. **Role Assignments**: Visualizations showing which grid cells are assigned to internal, blanket, or external roles

## Performance Metrics

The script calculates and reports:

- Accuracy of DMBD role assignments compared to ground truth
- Detailed diagnostics of tensor dimensions and model parameters
- Summary of DMBD results with shape information for all outputs

## Troubleshooting

### Tensor Shape Mismatch Issues

If you encounter tensor shape mismatch errors:

- Try reducing the grid size (`--grid-size 8`)
- Increase the number of convergence attempts (`--convergence-attempts 5`)
- Use the `--verbose` flag to see detailed error messages

### Memory Errors

For memory-related issues:

- Reduce the grid size and time steps
- Use more conservative role and hidden dimensions
- Run in a GPU-enabled environment if available

### Visualization Issues

If you encounter visualization errors:

- Ensure matplotlib is properly installed: `pip install matplotlib`
- Check that you have a working backend for animation generation

## References

For more information about the DMBD algorithm and its applications, see:

- The main README file in the repository root
- Documentation in the repository's docs directory
- The academic papers referenced in the code comments
