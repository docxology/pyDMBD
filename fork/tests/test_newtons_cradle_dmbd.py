"""
Test suite for Dynamic Markov Blanket Detection on Newton's Cradle.

This module provides tests for DMBD inference and visualization on the Newton's Cradle
system, which demonstrates interesting physical interactions and causal relationships.
Memory-efficient settings are used to prevent crashes.
"""
import os
import sys
import gc
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

# Create output directory for Newton's Cradle DMBD results
output_dir = Path(__file__).parent.parent / "dmbd_outputs"
cradle_dir = output_dir / "newtons_cradle"
os.makedirs(cradle_dir, exist_ok=True)

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_newtons_cradle_dmbd_memory_efficient(seed, caplog):
    """
    Test DMBD inference on Newton's Cradle with memory-efficient settings.
    
    This test addresses the 'Killed' issue in the original test by:
    1. Using smaller batch sizes and sequence lengths
    2. Limiting model complexity
    3. Manually collecting garbage during processing
    4. Processing in smaller chunks if needed
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Import Newton's Cradle simulation after setting seeds
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "simulations"))
        from newtonscradle import newtons_cradle
        
        caplog.set_level("INFO")
        print("Running Newton's Cradle simulation with memory-efficient DMBD...")
        
        # Check for quick test mode
        quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
        
        # Configure for a shorter simulation to avoid memory issues
        n_steps = 150 if quick_mode else 300  # Reduced from default
        n_balls = 3 if quick_mode else 5      # Fewer balls in quick mode
        
        print(f"Simulating Newton's Cradle with {n_balls} balls for {n_steps} steps...")
        
        # Run simulation with memory-efficient settings
        trajectory = newtons_cradle.simulate(
            batch_num=1, 
            n_steps=n_steps, 
            n_balls=n_balls,
            return_tensors=True
        )
        
        # Verify the output shape
        assert len(trajectory.shape) == 4, f"Expected 4D tensor, got shape {trajectory.shape}"
        print(f"Simulation shape: {trajectory.shape}")
        
        # Create a basic visualization of positions
        plt.figure(figsize=(12, 6))
        batch_idx = 0
        
        # Plot the x positions of each ball over time
        for i in range(n_balls):
            plt.plot(trajectory[:, batch_idx, i, 0].numpy(), 
                   label=f'Ball {i+1} position')
        
        plt.xlabel('Time steps')
        plt.ylabel('X position')
        plt.title("Newton's Cradle Simulation - Ball Positions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(cradle_dir / "newtons_cradle_positions.png")
        plt.close()
        
        # Force garbage collection to free memory
        gc.collect()
        
        # We'll use just the position data (first 2 features) for DMBD analysis
        pos_data = trajectory[:, :, :, :2]
        
        # If memory is a concern, can downsample the time dimension
        if not quick_mode and pos_data.shape[0] > 200:
            # Downsample to 200 time steps
            indices = torch.linspace(0, pos_data.shape[0]-1, 200).long()
            pos_data = pos_data[indices]
            print(f"Downsampled to {pos_data.shape[0]} time steps to save memory")
        
        # Ensure correct format for DMBD: [batch_size, seq_length, n_obs, obs_dim]
        data_reshaped = pos_data.permute(1, 0, 2, 3)
        
        # Define observation shape
        obs_shape = (n_balls, 2)  # n_balls objects, each with 2D position
        
        print(f"Running DMBD analysis on data with shape {data_reshaped.shape}...")
        
        # Use smaller dimensions and fewer iterations for better memory efficiency
        # For Newton's Cradle, we expect to discover the physical connections between the balls
        dmbd_results = run_dmbd_analysis(
            data_reshaped,
            obs_shape=obs_shape,
            role_dims=(2, 2, 2) if quick_mode else (3, 3, 3),  # Smaller dims for memory efficiency
            hidden_dims=(1, 1, 1) if quick_mode else (2, 2, 2),  # Smaller hidden dimensions
            number_of_objects=1,
            training_iters=8 if quick_mode else 15,  # Fewer iterations
            lr=0.6,  # Slightly higher learning rate for faster convergence
            name="newtons_cradle_memory_efficient"
        )
        
        # Force garbage collection again
        gc.collect()
        
        # Basic test assertions
        assert dmbd_results is not None, "DMBD analysis should return results"
        assert 'model' in dmbd_results, "DMBD results should include a model"
        
        # Create a custom visualization that's more memory-efficient
        if 'assignments' in dmbd_results:
            # Get model and assignments
            model = dmbd_results['model']
            assignments = dmbd_results['assignments']
            
            # Create a minimal but informative visualization
            plt.figure(figsize=(10, 8))
            
            plt.suptitle("Newton's Cradle - DMBD Analysis", fontsize=16)
            
            # Plot assignments over time (more memory efficient than drawing the cradle)
            plt.subplot(2, 1, 1)
            
            # Limit to a subset of time steps if still large
            max_time_steps = 100
            if assignments.shape[0] > max_time_steps:
                step = assignments.shape[0] // max_time_steps
                assignments_subset = assignments[::step]
                time_label = f"Time Step (every {step} steps)"
            else:
                assignments_subset = assignments
                time_label = "Time Step"
            
            plt.imshow(
                assignments_subset[:, 0, :].cpu().numpy().T,
                aspect='auto',
                cmap=ListedColormap(['red', 'green', 'blue']),
                norm=Normalize(vmin=0, vmax=2)
            )
            
            plt.colorbar(ticks=[0, 1, 2], 
                       label='Role Assignment (0:sensor, 1:boundary, 2:internal)')
            plt.xlabel(time_label)
            plt.ylabel('Ball')
            plt.yticks(range(n_balls), [f'Ball {i+1}' for i in range(n_balls)])
            plt.title("Role Assignments Over Time")
            
            # Plot statistics rather than full reconstruction
            plt.subplot(2, 1, 2)
            
            # Calculate percentage of time each ball spends in each role
            role_percentages = np.zeros((n_balls, 3))
            for ball in range(n_balls):
                for role in range(3):
                    role_percentages[ball, role] = (
                        (assignments[:, 0, ball].cpu().numpy() == role).mean() * 100
                    )
            
            # Plot as stacked bars
            bottom = np.zeros(n_balls)
            for role, color in enumerate(['red', 'green', 'blue']):
                plt.bar(
                    range(n_balls), 
                    role_percentages[:, role],
                    bottom=bottom, 
                    color=color,
                    label=f"{'Sensor' if role==0 else 'Boundary' if role==1 else 'Internal'}"
                )
                bottom += role_percentages[:, role]
            
            plt.xlabel('Ball')
            plt.ylabel('Percentage of Time')
            plt.title('Role Assignment Distribution')
            plt.legend()
            plt.xticks(range(n_balls), [f'Ball {i+1}' for i in range(n_balls)])
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(cradle_dir / "newtons_cradle_dmbd_analysis.png")
            plt.close()
            
            print("Newton's Cradle DMBD analysis completed successfully!")
        else:
            print("DMBD assignment analysis failed.")
        
        # Create a report of the findings
        with open(cradle_dir / "newtons_cradle_dmbd_report.txt", "w") as f:
            f.write("NEWTON'S CRADLE DMBD ANALYSIS REPORT\n")
            f.write("===================================\n\n")
            f.write(f"Simulation parameters:\n")
            f.write(f"- Number of balls: {n_balls}\n")
            f.write(f"- Number of time steps: {n_steps}\n")
            f.write(f"- Data shape: {data_reshaped.shape}\n\n")
            
            if dmbd_results and 'model' in dmbd_results:
                model = dmbd_results['model']
                f.write(f"DMBD model parameters:\n")
                f.write(f"- Role dimensions: {tuple(model.role_dims)}\n")
                f.write(f"- Hidden dimensions: {tuple(model.hidden_dims)}\n")
                
                if 'elbo_history' in dmbd_results:
                    elbo = dmbd_results['elbo_history']
                    f.write(f"- Final ELBO: {elbo[-1]:.4f}\n")
                    f.write(f"- ELBO improvement: {elbo[-1] - elbo[0]:.4f}\n\n")
                
                if 'assignments' in dmbd_results:
                    assignments = dmbd_results['assignments']
                    f.write("Role assignment summary:\n")
                    for ball in range(n_balls):
                        sensor_pct = (assignments[:, 0, ball].cpu().numpy() == 0).mean() * 100
                        boundary_pct = (assignments[:, 0, ball].cpu().numpy() == 1).mean() * 100
                        internal_pct = (assignments[:, 0, ball].cpu().numpy() == 2).mean() * 100
                        
                        f.write(f"Ball {ball+1}:\n")
                        f.write(f"  - Sensor role: {sensor_pct:.1f}%\n")
                        f.write(f"  - Boundary role: {boundary_pct:.1f}%\n")
                        f.write(f"  - Internal role: {internal_pct:.1f}%\n")
                        f.write(f"  - Dominant role: {'Sensor' if sensor_pct > max(boundary_pct, internal_pct) else 'Boundary' if boundary_pct > internal_pct else 'Internal'}\n\n")
            else:
                f.write("DMBD analysis failed to produce a valid model.\n")
        
    except ImportError as e:
        print(f"Could not import Newton's Cradle simulation: {e}")
        pytest.skip("Newton's Cradle simulation not available")
    except Exception as e:
        print(f"Error in test_newtons_cradle_dmbd_memory_efficient: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to mark the test as failed
        raise

if __name__ == "__main__":
    # This allows running the test directly
    pytest.main(["-xvs", __file__]) 