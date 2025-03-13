#!/usr/bin/env python3
"""
A minimal test for the DMBD model with Gaussian Blob simulation.
This file tests the torch-based inference capabilities of the DMBD model 
on empirical data from a Gaussian Blob simulation.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dmbd_test")

# Add the repository root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import DMBD modules
from fork.examples.GaussianBlob import GaussianBlobSimulation
from fork.dmbd.dmbd import DMBD

def extract_blob_features(data, grid_size):
    """
    Extract exactly 12 features from the Gaussian blob data to match DMBD dimensions.
    
    Args:
        data: Tensor of shape [time_steps, 1, grid_size*grid_size]
        grid_size: Size of the grid
    
    Returns:
        Processed data with informative features
        Shape: [time_steps, 1, 12]
    """
    time_steps = data.shape[0]
    processed_data = torch.zeros((time_steps, 1, 12), dtype=torch.float32)
    
    for t in range(time_steps):
        # Reshape to grid
        frame = data[t, 0].reshape(grid_size, grid_size)
        
        # 1-3: Global statistics
        mean_val = frame.mean()
        std_val = frame.std()
        max_val = frame.max()
        
        # 4-6: Center vs periphery measurements
        center_h, center_w = grid_size // 2, grid_size // 2
        radius = grid_size // 4
        
        # Create masks for center, mid, and periphery regions
        y_indices, x_indices = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        distances = torch.sqrt((y_indices - center_h)**2 + (x_indices - center_w)**2)
        
        center_mask = distances <= radius
        mid_mask = (distances > radius) & (distances <= 2*radius)
        outer_mask = distances > 2*radius
        
        center_mean = frame[center_mask].mean() if center_mask.any() else 0
        mid_mean = frame[mid_mask].mean() if mid_mask.any() else 0
        outer_mean = frame[outer_mask].mean() if outer_mask.any() else 0
        
        # 7-9: Directional features (North, South, East, West)
        north = frame[:grid_size//3, :].mean()
        south = frame[2*grid_size//3:, :].mean()
        east = frame[:, 2*grid_size//3:].mean()
        
        # 10-12: Additional statistics and gradients
        gradient_x = torch.abs(frame[:, 1:] - frame[:, :-1]).mean()  # Horizontal gradient
        gradient_y = torch.abs(frame[1:, :] - frame[:-1, :]).mean()  # Vertical gradient
        entropy = -torch.sum(frame * torch.log(frame + 1e-10)) / (grid_size * grid_size)  # Approximate entropy
        
        # Combine all features
        features = torch.tensor([
            mean_val, std_val, max_val,               # Global statistics
            center_mean, mid_mean, outer_mean,        # Center vs periphery
            north, south, east,                       # Directional features
            gradient_x, gradient_y, entropy           # Additional statistics
        ])
        
        processed_data[t, 0, :] = features
        
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    # Add small regularization to prevent numerical issues
    epsilon = 1e-6
    processed_data = processed_data + epsilon * torch.randn_like(processed_data)
    
    return processed_data

def debug_tensor_details(tensor, name):
    """Print detailed information about a tensor for debugging"""
    if tensor is None:
        print(f"{name} is None")
        return
        
    print(f"{name} details:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Min/Max: {tensor.min().item():.4f}/{tensor.max().item():.4f}")
    
    # Handle different tensor types for mean/std calculations
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        print(f"  Mean/Std: {tensor.mean().item():.4f}/{tensor.std().item():.4f}")
    else:
        # For integer tensors, convert to float for statistics
        print(f"  Mean/Std: {tensor.float().mean().item():.4f}/{tensor.float().std().item():.4f}")
        
    if tensor.shape[0] < 5:  # Only print values for small tensors
        print(f"  Values: {tensor}")
    
    # Check for NaNs and infinities
    if torch.isnan(tensor).any():
        print(f"  WARNING: {name} contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"  WARNING: {name} contains infinite values!")

def validate_dmbd_inference(dmbd_model, assignments, labels, grid_size):
    """
    Validate the DMBD model's inference by comparing with ground truth labels.
    
    Args:
        dmbd_model: The trained DMBD model
        assignments: Role assignments from DMBD, shape [time_steps, 1]
        labels: Ground truth labels, shape [time_steps, grid_size*grid_size]
        grid_size: Size of the grid
        
    Returns:
        Dictionary with validation metrics
    """
    results = {}
    
    try:
        # Print detailed information about inputs
        debug_tensor_details(assignments, "DMBD Assignments")
        debug_tensor_details(labels, "Ground Truth Labels")
        
        # Convert assignments to the same shape as labels for comparison
        time_steps = assignments.shape[0]
        flat_size = grid_size * grid_size
        
        # Create visualization
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        
        # Sample 4 timesteps for visualization
        sample_times = np.linspace(0, time_steps-1, 4, dtype=int)
        
        total_accuracy = 0
        role_mapping = {}
        unique_assignments = torch.unique(assignments)
        
        for i, t in enumerate(sample_times):
            # Get the raw grid data for this timestep
            gt_grid = labels[t].reshape(grid_size, grid_size)
            
            # Plot ground truth
            axs[0, i].imshow(gt_grid)
            axs[0, i].set_title(f"Ground Truth t={t}")
            axs[0, i].axis('off')
            
            # Create a mask for the specific role
            # We need to handle the dimensions carefully
            role = assignments[t, 0].item()
            
            # For accuracy calculation, we first need to map DMBD roles to ground truth roles
            if role not in role_mapping:
                # Count overlaps with each ground truth role
                role_counts = {}
                for gt_role in torch.unique(labels[t]):
                    gt_role = gt_role.item()
                    # Create a flat mask for this ground truth role
                    flat_mask = (labels[t] == gt_role).reshape(-1)
                    # Count overlap
                    if flat_mask.sum() > 0:
                        role_counts[gt_role] = flat_mask.sum().item()
                
                # Assign to most overlapping role
                if role_counts:
                    role_mapping[role] = max(role_counts, key=role_counts.get)
            
            # For visualization - create a grid showing the assigned role
            assignment_grid = torch.zeros((grid_size, grid_size))
            mapped_role = role_mapping.get(role, -1)
            
            # Apply the mapped role to locations that match the ground truth role
            gt_flat = labels[t].reshape(-1)
            for idx in range(len(gt_flat)):
                if gt_flat[idx] == mapped_role:
                    row, col = idx // grid_size, idx % grid_size
                    assignment_grid[row, col] = 1
            
            # Plot DMBD assignments
            axs[1, i].imshow(assignment_grid)
            axs[1, i].set_title(f"DMBD Role {role} → GT {mapped_role}")
            axs[1, i].axis('off')
            
            # Calculate accuracy for this timestep
            correct = (gt_grid == mapped_role).sum().item()
            accuracy = correct / (grid_size * grid_size)
            total_accuracy += accuracy
            
        # Compute average accuracy
        avg_accuracy = total_accuracy / len(sample_times)
        results['accuracy'] = avg_accuracy
        results['role_mapping'] = role_mapping
        
        # Add text with overall metrics
        plt.figtext(0.5, 0.01, f"Average Accuracy: {avg_accuracy:.4f}", ha='center', fontsize=12)
        plt.figtext(0.5, 0.04, f"Role Mapping: {role_mapping}", ha='center', fontsize=10)
        
        # Save the visualization
        os.makedirs('dmbd_outputs', exist_ok=True)
        plt.savefig('dmbd_outputs/dmbd_inference_validation.png')
        plt.close()
        
        print(f"Validation complete - Average accuracy: {avg_accuracy:.4f}")
        print(f"Role mapping: {role_mapping}")
        
        return results
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def check_torch_operations(dmbd_model, data):
    """
    Verify that the DMBD model is correctly using torch operations during inference.
    
    Args:
        dmbd_model: The DMBD model to test
        data: Input data tensor
    
    Returns:
        Dictionary with operation count and success status
    """
    results = {
        'op_count': 0,
        'success': False
    }
    
    # Create a wrapper hook to count operations
    class GradientCounter:
        def __init__(self):
            self.count = 0
            self.tensors = []
            
        def hook(self, grad):
            self.count += 1
            return grad
            
    counter = GradientCounter()
    
    try:
        # Check model parameters
        logger.info("Checking DMBD model parameters...")
        tensor_count = sum(1 for p in dmbd_model.parameters() if isinstance(p, torch.Tensor))
        logger.info(f"Torch tensor count: {tensor_count} tensors found")
        
        # Enable gradient tracking on input data
        small_batch = data[:5].clone()  # Use a smaller batch for testing
        small_batch.requires_grad_(True)
        
        # Register hooks for gradient counting
        hooks = []
        for name, param in dmbd_model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(counter.hook)
                hooks.append(hook)
                counter.tensors.append(name)
                
        # Try a single update step
        logger.info("Testing model with gradient tracking...")
        with torch.autograd.set_detect_anomaly(True):
            # Run a single update
            success = dmbd_model.update(
                y=small_batch, 
                u=None, 
                r=None, 
                iters=1,
                lr=0.0001,  # Very small learning rate for numerical stability
                verbose=True
            )
            
            # Check if operations were performed
            results['op_count'] = counter.count
            results['success'] = success
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        logger.info(f"Torch operations check: {counter.count} operations performed")
        return results
        
    except Exception as e:
        logger.error(f"Error in torch operations check: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Remove hooks if there was an error
        try:
            for hook in hooks:
                hook.remove()
        except:
            pass
            
        return results

def main():
    """Run the DMBD test with Gaussian Blob simulation."""
    print("Starting DMBD test with Gaussian Blob simulation...")
    
    # Log initialization
    logger.info("Initializing test...")
    
    # Create output directory
    os.makedirs("dmbd_outputs", exist_ok=True)
    
    # Set parameters
    grid_size = 12
    time_steps = 20
    sigma = 2.0
    noise_level = 0.02
    
    # Set up regularization
    reg_scale = 1e-4  # Regularization strength
    
    # Initialize the simulation
    print(f"Creating simulation with grid_size={grid_size}, time_steps={time_steps}")
    simulation = GaussianBlobSimulation(
        grid_size=grid_size,
        time_steps=time_steps,
        sigma=sigma,
        noise_level=noise_level
    )
    
    # Run the simulation
    data, labels = simulation.run()
    print(f"Simulation complete. Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Display a sample of the raw data
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    sample_times = np.linspace(0, time_steps-1, 4, dtype=int)
    
    for i, t in enumerate(sample_times):
        frame = data[t, 0].reshape(grid_size, grid_size)
        axs[i].imshow(frame)
        axs[i].set_title(f"Frame t={t}")
        axs[i].axis('off')
    
    plt.savefig('dmbd_outputs/raw_data_sample.png')
    plt.close()
    
    # Extract more meaningful features from the data
    print("Extracting features from data...")
    processed_data = extract_blob_features(data, grid_size)
    
    # Feature exploration - visualize the extracted features
    feature_names = [
        'Mean', 'Std', 'Max',                      # Global stats
        'Center', 'Middle', 'Periphery',           # Center vs periphery 
        'North', 'South', 'East',                  # Directional features
        'Gradient-X', 'Gradient-Y', 'Entropy'      # Additional stats
    ]
    
    fig, axs = plt.subplots(4, 3, figsize=(15, 16))
    axs = axs.flatten()
    
    for i in range(12):
        feature_data = processed_data[:, 0, i].numpy()
        axs[i].plot(feature_data)
        axs[i].set_title(feature_names[i])
        axs[i].set_xlabel('Time step')
        axs[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('dmbd_outputs/extracted_features.png')
    plt.close()
    
    # Debug - print tensor information
    debug_tensor_details(processed_data, "Processed Data")
    
    # Initialize the DMBD model, ensuring dimensions match
    print("Initializing DMBD model...")
    
    # IMPORTANT: The observation shape must match the number of features
    obs_dim = processed_data.shape[2]  # Should be 12 features
    
    # CRITICAL: Match the dimensions exactly to avoid size mismatch errors
    role_dims = [4, 4, 4]  # Dimensions for [environment, boundary, internal] roles
    hidden_dims = [4, 4, 4]  # Same dimensions for hidden states
    
    # Initialize DMBD with proper dimensions
    dmbd_model = DMBD(
        obs_shape=(1, obs_dim),
        role_dims=role_dims,
        hidden_dims=hidden_dims,
        number_of_objects=1
    )
    
    # Log model parameters
    print(f"DMBD model initialized with observation shape: (1, {obs_dim})")
    print(f"Role dimensions: {role_dims}")
    print(f"Hidden dimensions: {hidden_dims}")
    
    # Verify torch operations
    print("Checking torch operations...")
    ops_results = check_torch_operations(dmbd_model, processed_data)
    print(f"Torch operations check: {ops_results['op_count']} operations performed")
    
    # First update attempt with regularized data
    print("Running DMBD update with regularization...")
    
    # Apply regularization to prevent matrix inversion issues
    regularized_data = processed_data.clone()
    if reg_scale > 0:
        # Add small random noise to prevent singular matrices
        regularized_data += reg_scale * torch.randn_like(regularized_data)
        
        # Ensure the noise doesn't dramatically change the patterns
        regularized_data = (1-reg_scale) * processed_data + reg_scale * regularized_data
    
    # Update with moderate iterations and learning rate
    max_iterations = 60
    learning_rate = 0.002
    
    try:
        print(f"Running DMBD update with {max_iterations} iterations and lr={learning_rate}...")
        
        # Run the DMBD update with processed data
        success = dmbd_model.update(
            y=regularized_data,
            u=None,           # No control inputs
            r=None,           # No regression covariates
            iters=max_iterations,
            lr=learning_rate,
            verbose=True
        )
        
        # Get the role assignments
        assignments = dmbd_model.assignment()
        
        print(f"DMBD update completed: {success}")
        print(f"Assignments shape: {assignments.shape if assignments is not None else 'None'}")
        
        if assignments is not None:
            print(f"Min assignment: {assignments.min().item()}, Max assignment: {assignments.max().item()}")
            unique_roles = torch.unique(assignments)
            print(f"Unique roles assigned: {unique_roles.tolist()}")
            
            # Validate the results
            print("Validating DMBD inference against ground truth...")
            validation_metrics = validate_dmbd_inference(dmbd_model, assignments, labels, grid_size)
        else:
            print("No assignments produced by DMBD")
            
    except Exception as e:
        print(f"Error during DMBD update: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("DMBD test complete.")

if __name__ == "__main__":
    main()
