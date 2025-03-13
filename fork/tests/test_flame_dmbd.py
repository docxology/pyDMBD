"""
Test suite for Dynamic Markov Blanket Detection on the Flame Propagation system.

This module provides tests for DMBD inference and visualization on a flame
propagation simulation, demonstrating emergent structures in reaction-diffusion systems.
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

# Create output directory for Flame DMBD results
output_dir = Path(__file__).parent.parent / "dmbd_outputs"
flame_dir = output_dir / "flame"
os.makedirs(flame_dir, exist_ok=True)

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_flame_dmbd_basic(seed, caplog):
    """Test basic DMBD inference on the Flame Propagation system."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Import from examples folder
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    
    caplog.set_level("INFO")
    print("Running Flame Propagation simulation with basic DMBD inference...")
    
    try:
        # Check for quick test mode
        quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
        
        # Import FlameSimulator and generate data
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from flame import FlameSimulator
        
        # Simulation parameters - use smaller values in quick mode
        num_steps = 1000 if quick_mode else 2000
        delta_t = 0.005
        thermal_diffusivity = 0.5
        temperature_threshold = 0.4 + 0.1 * torch.rand(1)
        num_sources = 20 if quick_mode else 50
        num_batches = 2 if quick_mode else 5
        num_x = 200 if quick_mode else 1000
        
        print(f"Generating Flame Propagation data with {num_steps} steps and {num_batches} batches...")
        
        # Create flame simulator
        simulator = FlameSimulator(
            num_steps=num_steps, 
            delta_t=delta_t, 
            thermal_diffusivity=thermal_diffusivity, 
            temperature_threshold=temperature_threshold, 
            num_sources=num_sources
        )
        
        # Run simulation
        temperature, ignition_times, heat_released = simulator.simulate()
        
        # Verify the output shape
        print(f"Temperature shape: {temperature.shape}")
        print(f"Ignition times shape: {ignition_times.shape}")
        
        # Create a basic visualization of the flame propagation
        plt.figure(figsize=(14, 8))
        
        # Plot temperature profiles at different times
        times = torch.linspace(100, temperature.shape[0]-100, 4).long()
        plt.subplot(2, 1, 1)
        plt.plot(temperature[times, :].T, linewidth=2)
        plt.xlabel('Position')
        plt.ylabel('Temperature')
        plt.title('Flame Temperature Profiles at Different Times')
        plt.grid(True)
        
        # Plot ignition times
        plt.subplot(2, 1, 2)
        valid_times = ignition_times[~torch.isinf(ignition_times)]
        plt.plot(valid_times, 'r-')
        plt.xlabel('Source Index')
        plt.ylabel('Ignition Time')
        plt.title('Ignition Times for Each Heat Source')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(flame_dir / "flame_basic_visualization.png")
        plt.close()
        
        # Generate fine-grained data for DMBD analysis
        fine_temp, fuel, ox, source_locations = simulator.fine_grain(num_x)
        
        # Create a heatmap of the temperature evolution
        plt.figure(figsize=(12, 8))
        plt.imshow(
            fine_temp.T, 
            cmap='hot', 
            origin='lower',
            aspect='auto',
            extent=[0, fine_temp.shape[0]*delta_t, 0, fine_temp.shape[1]]
        )
        plt.colorbar(label='Temperature')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Flame Propagation Heatmap')
        plt.savefig(flame_dir / "flame_heatmap.png")
        plt.close()
        
        # Prepare data for DMBD analysis
        # We'll use the fine-grained temperature, fuel, and oxidizer data
        # Shape: [timesteps, positions]
        
        # Select a subset of the data to make computation more manageable
        subsample_timesteps = 100 if quick_mode else 200
        subsample_positions = 100 if quick_mode else 200
        
        # Create indices for subsampling
        time_indices = torch.linspace(0, fine_temp.shape[0]-1, subsample_timesteps).long()
        pos_indices = torch.linspace(0, fine_temp.shape[1]-1, subsample_positions).long()
        
        # Extract subsampled data
        subsampled_temp = fine_temp[time_indices][:, pos_indices]
        subsampled_fuel = fuel[time_indices][:, pos_indices]
        subsampled_ox = ox[time_indices][:, pos_indices]
        
        print(f"Subsampled data shape: {subsampled_temp.shape}")
        
        # Create a combined dataset with temperature, fuel, and oxidizer
        # Shape: [timesteps, positions, 3]
        combined_data = torch.stack([subsampled_temp, subsampled_fuel, subsampled_ox], dim=-1)
        
        # Reshape for DMBD: [batch_size, seq_length, n_obs, obs_dim]
        # We'll treat each position as a separate batch
        batch_size = combined_data.shape[1]  # positions
        seq_length = combined_data.shape[0]  # timesteps
        n_obs = combined_data.shape[2]       # features (temp, fuel, ox)
        obs_dim = 1
        
        # Transpose and add dimension for obs_dim
        data_reshaped = combined_data.transpose(0, 1).unsqueeze(-1)
        
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
            name="flame_basic",
            visualize_timeseries=True
        )
        
        # Basic test assertions
        assert dmbd_results is not None, "DMBD analysis should return results"
        assert os.path.exists(flame_dir / "flame_basic_markov_blanket.png"), "DMBD should generate Markov blanket visualization"
        assert os.path.exists(flame_dir / "flame_basic_assignments.png"), "DMBD should generate role assignments visualization"
        
        print("Flame Propagation DMBD analysis completed successfully")
        
    except Exception as e:
        print(f"Error in flame propagation DMBD analysis: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Flame propagation DMBD analysis failed: {e}")

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_flame_dmbd_hyperparameter_search(seed, caplog):
    """Test DMBD on Flame Propagation with hyperparameter search for optimal configuration."""
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
    print("Running Flame Propagation simulation with DMBD hyperparameter search...")
    
    try:
        # Import FlameSimulator and generate data
        sys.path.append(str(Path(__file__).parent.parent / "examples"))
        from flame import FlameSimulator
        
        # Simulation parameters - use smaller values in quick mode
        num_steps = 500 if quick_mode else 1000
        delta_t = 0.005
        thermal_diffusivity = 0.5
        temperature_threshold = 0.4 + 0.1 * torch.rand(1)
        num_sources = 20 if quick_mode else 30
        num_x = 100 if quick_mode else 200
        
        print(f"Generating Flame Propagation data with {num_steps} steps...")
        
        # Create flame simulator
        simulator = FlameSimulator(
            num_steps=num_steps, 
            delta_t=delta_t, 
            thermal_diffusivity=thermal_diffusivity, 
            temperature_threshold=temperature_threshold, 
            num_sources=num_sources
        )
        
        # Run simulation
        temperature, ignition_times, heat_released = simulator.simulate()
        
        # Generate fine-grained data
        fine_temp, fuel, ox, source_locations = simulator.fine_grain(num_x)
        
        # Prepare data for DMBD analysis - smaller subsets for hyperparameter search
        subsample_timesteps = 50 if quick_mode else 100
        subsample_positions = 50 if quick_mode else 100
        
        # Create indices for subsampling
        time_indices = torch.linspace(0, fine_temp.shape[0]-1, subsample_timesteps).long()
        pos_indices = torch.linspace(0, fine_temp.shape[1]-1, subsample_positions).long()
        
        # Extract subsampled data
        subsampled_temp = fine_temp[time_indices][:, pos_indices]
        subsampled_fuel = fuel[time_indices][:, pos_indices]
        subsampled_ox = ox[time_indices][:, pos_indices]
        
        # Create a combined dataset
        combined_data = torch.stack([subsampled_temp, subsampled_fuel, subsampled_ox], dim=-1)
        
        # Reshape for DMBD
        data_reshaped = combined_data.transpose(0, 1).unsqueeze(-1)
        
        # Define observation shape
        obs_shape = (combined_data.shape[2], 1)
        
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
                name=f"flame_config{i+1}"
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
        plt.title("Flame Propagation DMBD - Model Comparison")
        
        # Collect all ELBO histories
        for i in range(len(role_dims_options)):
            config_file = flame_dir / f"flame_config{i+1}_dmbd_results.pt"
            if os.path.exists(config_file):
                results = torch.load(config_file)
                if 'elbo_history' in results:
                    plt.plot(results['elbo_history'], 
                           label=f"Config {i+1}: {role_dims_options[i]}/{hidden_dims_options[i]}")
        
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(flame_dir / "flame_dmbd_comparison.png")
        plt.close()
        
        print("Flame Propagation DMBD hyperparameter search completed successfully")
        
    except Exception as e:
        print(f"Error in flame propagation DMBD hyperparameter search: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Flame propagation DMBD hyperparameter search failed: {e}")

if __name__ == "__main__":
    # This allows running the test directly
    pytest.main(["-xvs", __file__]) 