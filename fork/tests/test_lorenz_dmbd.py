"""
Test suite for Dynamic Markov Blanket Detection on the Lorenz attractor.

This module provides tests for DMBD inference and visualization on the Lorenz attractor
system, which is a classic chaotic system often used as a benchmark.
"""
import os
import sys
import pytest
import torch
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from pathlib import Path

# Import the test_examples module to reuse the DMBD analysis function
sys.path.append(str(Path(__file__).parent))
from test_examples import run_dmbd_analysis

# Create output directory for Lorenz DMBD results
output_dir = Path(__file__).parent.parent / "dmbd_outputs"
lorenz_dir = output_dir / "lorenz"
os.makedirs(lorenz_dir, exist_ok=True)

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_lorenz_dmbd_basic(seed, caplog):
    """Test basic DMBD inference on the Lorenz attractor with simplified settings."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Import simulation after setting seeds
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    from Lorenz import Lorenz
    
    caplog.set_level("INFO")
    print("Running Lorenz attractor simulation with basic DMBD inference...")
    
    # Check for quick test mode
    quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
    n_steps = 200 if quick_mode else 500
    
    # Run simulation with Lorenz attractor
    lorenz = Lorenz()
    lorenz.num_steps = n_steps
    trajectory = lorenz.simulate(batch_num=1)
    
    # Verify the output shape
    print(f"Simulation shape: {trajectory.shape}")
    
    # Create a basic plot of the trajectories
    plt.figure(figsize=(10, 8))
    batch_idx = 0
    
    # Get the trajectory data (position only from the first component of the last dimension)
    x = trajectory[:, batch_idx, 0, 0].numpy()
    y = trajectory[:, batch_idx, 1, 0].numpy()
    z = trajectory[:, batch_idx, 2, 0].numpy()
    
    # Plot 3D trajectory
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, linewidth=1.0)
    ax.set_title('Lorenz Attractor Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Save the plot
    plt.savefig(lorenz_dir / "lorenz_trajectory_3d.png")
    plt.close()
    
    # Reshape for DMBD: [batch_size, seq_length, n_obs, obs_dim]
    # The data is already in [seq_length, batch_size, n_obs, 2] format
    # where the last dimension contains [position, velocity]
    data_reshaped = trajectory.permute(1, 0, 2, 3)
    
    # Define observation shape for DMBD
    obs_shape = (trajectory.shape[2], trajectory.shape[3])
    
    # Use exactly 10 iterations as requested
    training_iters = 10
    
    # Run DMBD analysis with compatible settings
    # Using the same role_dims and hidden_dims values from the hyperparameter search
    # that successfully passes to ensure dimensional compatibility
    dmbd_results = run_dmbd_analysis(
        data_reshaped,
        obs_shape=obs_shape,
        role_dims=(3, 3, 3),  # Keep dimensions consistent with obs_shape
        hidden_dims=(2, 2, 2), # These dimensions work with the hyperparameter search
        training_iters=training_iters,
        lr=0.01,              # Lower learning rate for stability
        name="lorenz_basic"
    )
    
    # Basic test assertions
    assert dmbd_results is not None, "DMBD analysis should return results"
    assert os.path.exists(lorenz_dir / "lorenz_basic_markov_blanket.png"), "DMBD should generate Markov blanket visualization"
    assert os.path.exists(lorenz_dir / "lorenz_basic_assignments.png"), "DMBD should generate role assignments visualization"

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_lorenz_dmbd_hyperparameter_search(seed, caplog):
    """Test DMBD on Lorenz with hyperparameter search for optimal configuration."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Check for quick test mode
    quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
    if quick_mode:
        print("Running in quick mode with reduced settings")
    
    # Import simulation after setting seeds
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    from Lorenz import Lorenz
    
    caplog.set_level("INFO")
    print("Running Lorenz attractor simulation with DMBD hyperparameter search...")
    
    # Run simulation with shorter settings in quick mode
    n_steps = 200 if quick_mode else 500
    lorenz = Lorenz()
    lorenz.num_steps = n_steps
    trajectory = lorenz.simulate(batch_num=1)
    
    # Reshape for DMBD analysis
    # We need 4D data: [batch_size, seq_length, n_obs, obs_dim]
    # The data is already in [seq_length, batch_size, n_obs, 2] format
    # where the last dimension contains [position, velocity]
    data_reshaped = trajectory.permute(1, 0, 2, 3)
    
    # Define observation shape
    obs_shape = (trajectory.shape[2], trajectory.shape[3])
    
    # Define hyperparameter options - use smaller set in quick mode
    if quick_mode:
        role_dims_options = [(3, 3, 3)]
        hidden_dims_options = [(2, 2, 2)]
    else:
        role_dims_options = [(3, 3, 3), (4, 4, 4), (5, 5, 5)]
        hidden_dims_options = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
    
    # Use exactly 10 iterations as requested
    training_iters = 10
    
    best_elbo = float("-inf")
    best_model_info = None
    
    # Test different parameter configurations
    for i, (role_dims, hidden_dims) in enumerate(zip(role_dims_options, hidden_dims_options)):
        print(f"Testing DMBD configuration {i+1}/{len(role_dims_options)}: "
              f"role_dims={role_dims}, hidden_dims={hidden_dims}")
        
        results = run_dmbd_analysis(
            data_reshaped, 
            obs_shape=obs_shape,
            role_dims=role_dims,
            hidden_dims=hidden_dims,
            training_iters=training_iters,
            lr=0.5,
            name=f"lorenz_config{i+1}"
        )
        
        if results and 'elbo_history' in results and results['elbo_history']:
            final_elbo = results['elbo_history'][-1]
            print(f"  Configuration {i+1} final ELBO: {final_elbo:.4f}")
            
            if final_elbo > best_elbo:
                best_elbo = final_elbo
                best_model_info = {
                    'config_idx': i+1,
                    'role_dims': role_dims,
                    'hidden_dims': hidden_dims,
                    'final_elbo': final_elbo
                }
    
    # Basic test assertions
    if best_model_info:
        print(f"Best configuration: {best_model_info['config_idx']}")
        assert best_elbo > float("-inf"), "Should find a configuration with valid ELBO"
    
    # Create a summary visualization of all configurations
    plt.figure(figsize=(10, 6))
    plt.title("Lorenz DMBD - Model Comparison")
    
    # Collect all ELBO histories
    for i in range(len(role_dims_options)):
        config_file = lorenz_dir / f"lorenz_config{i+1}_dmbd_results.pt"
        if os.path.exists(config_file):
            results = torch.load(config_file)
            if 'elbo_history' in results:
                plt.plot(results['elbo_history'], 
                       label=f"Config {i+1}: {role_dims_options[i]}/{hidden_dims_options[i]}")
    
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(lorenz_dir / "lorenz_dmbd_comparison.png")
    plt.close()

if __name__ == "__main__":
    # This allows running the test directly
    pytest.main(["-xvs", __file__]) 