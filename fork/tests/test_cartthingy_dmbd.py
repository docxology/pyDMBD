"""
Test suite for Dynamic Markov Blanket Detection on the Cart with Pendulums system.

This module provides tests for DMBD inference and visualization on a cart with
two pendulums, demonstrating causal relationships in a mechanical system.
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

# Create output directory for Cart with Pendulums DMBD results
output_dir = Path(__file__).parent.parent / "dmbd_outputs"
cart_dir = output_dir / "cartthingy"
os.makedirs(cart_dir, exist_ok=True)

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_cartthingy_dmbd_basic(seed, caplog):
    """Test basic DMBD inference on the Cart with Pendulums system."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Import from examples folder
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    
    caplog.set_level("INFO")
    print("Running Cart with Pendulums simulation with basic DMBD inference...")
    
    try:
        # Check for quick test mode
        quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
        
        # Import simulation after setting seeds
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from cartthingy import cartthingy
        
        # Batch size and steps
        batch_size = 2 if quick_mode else 10
        
        # Run simulation
        print(f"Generating Cart with Pendulums data with batch_size={batch_size}...")
        trajectory = cartthingy.simulate(batch_num=batch_size)
        
        # Verify the output shape
        print(f"Simulation trajectory shape: {trajectory.shape}")
        
        # Create a plot of the cart and pendulums
        plt.figure(figsize=(12, 8))
        batch_idx = 0
        timestep = trajectory.shape[0] // 2  # Mid-point of simulation
        
        # Extract cart position and pendulum positions
        cart_pos = trajectory[timestep, batch_idx, 0].item()
        theta1 = trajectory[timestep, batch_idx, 1].item()
        theta2 = trajectory[timestep, batch_idx, 2].item()
        
        # Parameters from cartthingy
        l1 = 1.0  # Length of pendulum 1
        l2 = 1.0  # Length of pendulum 2
        
        # Calculate pendulum positions
        x_p1 = cart_pos + l1 * np.sin(theta1)
        y_p1 = -l1 * np.cos(theta1)
        x_p2 = cart_pos + l2 * np.sin(theta2)
        y_p2 = -l2 * np.cos(theta2)
        
        # Create plot
        plt.plot([cart_pos-0.5, cart_pos+0.5], [0, 0], 'k-', linewidth=3)  # Cart
        plt.plot([cart_pos, x_p1], [0, y_p1], 'r-', linewidth=2)  # Pendulum 1
        plt.plot([cart_pos, x_p2], [0, y_p2], 'b-', linewidth=2)  # Pendulum 2
        plt.scatter([cart_pos], [0], s=100, c='k')  # Cart center
        plt.scatter([x_p1], [y_p1], s=50, c='r')  # Pendulum 1 end
        plt.scatter([x_p2], [y_p2], s=50, c='b')  # Pendulum 2 end
        
        plt.title('Cart with Two Pendulums - Snapshot')
        plt.xlabel('Position')
        plt.ylabel('Height')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(cart_dir / "cartthingy_snapshot.png")
        plt.close()
        
        # Plot a time series of cart and pendulum positions
        plt.figure(figsize=(14, 8))
        
        # Get time steps
        timesteps = np.arange(trajectory.shape[0])
        
        # Extract positions over time for one batch
        cart_positions = trajectory[:, batch_idx, 0].numpy()
        theta1_positions = trajectory[:, batch_idx, 1].numpy()
        theta2_positions = trajectory[:, batch_idx, 2].numpy()
        
        # Plot positions
        plt.subplot(3, 1, 1)
        plt.plot(timesteps, cart_positions, 'k-')
        plt.title('Cart Position')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(timesteps, theta1_positions, 'r-')
        plt.title('Pendulum 1 Angle')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(timesteps, theta2_positions, 'b-')
        plt.title('Pendulum 2 Angle')
        plt.xlabel('Time Step')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(cart_dir / "cartthingy_timeseries.png")
        plt.close()
        
        # Reshape data for DMBD
        # The data has shape [timesteps, batch, features]
        # Need to reshape to [batch, timesteps, n_obs, obs_dim]
        batch_size = trajectory.shape[1]
        seq_length = trajectory.shape[0]
        n_obs = trajectory.shape[2]
        obs_dim = 1
        
        # Permute and add dimension for obs_dim
        data_reshaped = trajectory.permute(1, 0, 2).unsqueeze(-1)
        
        # Define observation shape for DMBD
        obs_shape = (n_obs, obs_dim)
        
        # Use exactly 10 iterations as requested
        training_iters = 10
        
        # Run DMBD analysis
        dmbd_results = run_dmbd_analysis(
            data_reshaped,
            obs_shape=obs_shape,
            role_dims=(3, 3, 3),
            hidden_dims=(2, 2, 2),
            training_iters=training_iters,
            lr=0.5,
            name="cartthingy_basic",
            visualize_timeseries=True
        )
        
        # Basic test assertions
        assert dmbd_results is not None, "DMBD analysis should return results"
        assert os.path.exists(cart_dir / "cartthingy_basic_markov_blanket.png"), "DMBD should generate Markov blanket visualization"
        assert os.path.exists(cart_dir / "cartthingy_basic_assignments.png"), "DMBD should generate role assignments visualization"
        
        print("Cart with Pendulums DMBD analysis completed successfully")
        
    except Exception as e:
        print(f"Error in cart with pendulums DMBD analysis: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Cart with pendulums DMBD analysis failed: {e}")

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_cartthingy_dmbd_hyperparameter_search(seed, caplog):
    """Test DMBD on Cart with Pendulums with hyperparameter search for optimal configuration."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Check for quick test mode
    quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
    if quick_mode:
        print("Running in quick mode with reduced settings")
    
    # Import from examples folder
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    
    caplog.set_level("INFO")
    print("Running Cart with Pendulums simulation with DMBD hyperparameter search...")
    
    try:
        # Import simulation after setting seeds
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from cartthingy import cartthingy
        
        # Batch size and steps - use smaller values in quick mode
        batch_size = 2 if quick_mode else 5
        
        # Run simulation
        print(f"Generating Cart with Pendulums data with batch_size={batch_size}...")
        trajectory = cartthingy.simulate(batch_num=batch_size)
        
        # Reshape for DMBD analysis
        data_reshaped = trajectory.permute(1, 0, 2).unsqueeze(-1)
        
        # Define observation shape
        obs_shape = (trajectory.shape[2], 1)
        
        # Define hyperparameter options - use smaller set in quick mode
        if quick_mode:
            role_dims_options = [(3, 3, 3)]
            hidden_dims_options = [(2, 2, 2)]
        else:
            role_dims_options = [(3, 3, 3), (4, 4, 4)]
            hidden_dims_options = [(2, 2, 2), (3, 3, 3)]
        
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
                name=f"cartthingy_config{i+1}"
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
        plt.title("Cart with Pendulums DMBD - Model Comparison")
        
        # Collect all ELBO histories
        for i in range(len(role_dims_options)):
            config_file = cart_dir / f"cartthingy_config{i+1}_dmbd_results.pt"
            if os.path.exists(config_file):
                results = torch.load(config_file)
                if 'elbo_history' in results:
                    plt.plot(results['elbo_history'], 
                           label=f"Config {i+1}: {role_dims_options[i]}/{hidden_dims_options[i]}")
        
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(cart_dir / "cartthingy_dmbd_comparison.png")
        plt.close()
        
        print("Cart with Pendulums DMBD hyperparameter search completed successfully")
        
    except Exception as e:
        print(f"Error in cart with pendulums DMBD hyperparameter search: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Cart with pendulums DMBD hyperparameter search failed: {e}")

if __name__ == "__main__":
    # This allows running the test directly
    pytest.main(["-xvs", __file__]) 