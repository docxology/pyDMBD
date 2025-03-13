"""
Stabilized utilities for DMBD analysis of Gaussian blob data.
"""

import torch
import numpy as np
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def extract_features(data, grid_size, method="basic"):
    """
    Extract features from raw Gaussian Blob data using different methods.
    
    Args:
        data: Raw Gaussian Blob data with shape [time_steps, channels, grid_size*grid_size]
        grid_size: Size of the grid
        method: Feature extraction method to use
            - "basic": Just mean, std, max (3 features)
            - "spatial": Basic + center/periphery features (6 features)
            - "roles": Features designed to differentiate internal/blanket/external (9 features)
    
    Returns:
        Tensor with features extracted according to the specified method
    """
    time_steps, channels, _ = data.shape
    
    if method == "basic":
        # Simple feature extraction (3 features)
        result = torch.zeros((time_steps, channels, 3), dtype=torch.float32)
        
        for t in range(time_steps):
            # Calculate basic statistics
            result[t, 0, 0] = data[t, 0].mean()  # Global mean
            result[t, 0, 1] = data[t, 0].std()   # Global std
            result[t, 0, 2] = data[t, 0].max()   # Global max
            
        return result
    
    elif method == "spatial":
        # Spatial feature extraction (6 features)
        result = torch.zeros((time_steps, channels, 6), dtype=torch.float32)
        
        for t in range(time_steps):
            # Reshape to grid
            grid_data = data[t, 0].reshape(grid_size, grid_size)
            
            # Basic statistics
            result[t, 0, 0] = grid_data.mean()
            result[t, 0, 1] = grid_data.std()
            result[t, 0, 2] = grid_data.max()
            
            # Center region features
            center_h, center_w = grid_size // 2, grid_size // 2
            center_size = max(1, grid_size // 3)
            h_start, h_end = center_h - center_size, center_h + center_size + 1
            w_start, w_end = center_w - center_size, center_w + center_size + 1
            
            # Ensure indices are within bounds
            h_start, h_end = max(0, h_start), min(grid_size, h_end)
            w_start, w_end = max(0, w_start), min(grid_size, w_end)
            
            center_region = grid_data[h_start:h_end, w_start:w_end]
            result[t, 0, 3] = center_region.mean()  # Center mean
            
            # Periphery (everything outside center)
            mask = torch.ones_like(grid_data)
            mask[h_start:h_end, w_start:w_end] = 0
            periphery = grid_data * mask
            periphery_mean = periphery.sum() / max(mask.sum(), 1)
            result[t, 0, 4] = periphery_mean  # Periphery mean
            
            # Center to periphery ratio (high for blob center)
            result[t, 0, 5] = result[t, 0, 3] / max(periphery_mean, 1e-5)
            
        return result
    
    elif method == "roles":
        # Features designed to match internal/blanket/external roles (9 features)
        result = torch.zeros((time_steps, channels, 9), dtype=torch.float32)
        
        for t in range(time_steps):
            # Reshape to grid
            grid_data = data[t, 0].reshape(grid_size, grid_size)
            
            # Global features
            result[t, 0, 0] = grid_data.mean()
            result[t, 0, 1] = grid_data.std()
            result[t, 0, 2] = grid_data.max()
            
            # Internal region (high intensity)
            internal_mask = grid_data > 0.7
            if internal_mask.sum() > 0:
                result[t, 0, 3] = grid_data[internal_mask].mean()  # Internal mean
                result[t, 0, 4] = grid_data[internal_mask].std()   # Internal std
            else:
                result[t, 0, 3] = 0.0
                result[t, 0, 4] = 0.0
            
            # Blanket region (medium intensity)
            blanket_mask = (grid_data > 0.3) & (grid_data <= 0.7)
            if blanket_mask.sum() > 0:
                result[t, 0, 5] = grid_data[blanket_mask].mean()  # Blanket mean
                result[t, 0, 6] = grid_data[blanket_mask].std()   # Blanket std
            else:
                result[t, 0, 5] = 0.0
                result[t, 0, 6] = 0.0
            
            # External region (low intensity)
            external_mask = grid_data <= 0.3
            if external_mask.sum() > 0:
                result[t, 0, 7] = grid_data[external_mask].mean()  # External mean
                result[t, 0, 8] = grid_data[external_mask].std()   # External std
            else:
                result[t, 0, 7] = 0.0
                result[t, 0, 8] = 0.0
                
        return result
    
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

def evaluate_role_assignment(assignments, labels, grid_size):
    """
    Evaluate how well DMBD role assignments match the ground truth labels.
    
    Args:
        assignments: DMBD role assignments tensor
        labels: Ground truth labels tensor from GaussianBlob
        grid_size: Size of the grid
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {}
    
    # Print shapes for debugging
    logger.info(f"evaluate_role_assignment - Assignments shape: {assignments.shape}")
    logger.info(f"evaluate_role_assignment - Labels shape: {labels.shape}")
    
    # Handle different tensor dimensions
    if len(assignments.shape) == 3:  # [time_steps, channels, features]
        assignments = assignments[:, 0, :]
    
    # Handle case where assignments has shape [time_steps, 1]
    # and labels has shape [time_steps, grid_size**2]
    if assignments.shape[1] == 1 and labels.shape[1] == grid_size**2:
        # Broadcast the single assignment value to the whole grid for each timestep
        broadcasted_assignments = torch.zeros_like(labels)
        for t in range(assignments.shape[0]):
            broadcasted_assignments[t, :] = assignments[t, 0]
        assignments = broadcasted_assignments
    
    # Get unique values in assignments and labels
    unique_assignments = torch.unique(assignments).tolist()
    unique_labels = torch.unique(labels).tolist()
    
    # Map DMBD roles to ground truth roles
    role_mapping = {}
    for dmbd_role in unique_assignments:
        role_counts = {}
        for gt_role in unique_labels:
            # Count overlaps between this DMBD role and ground truth role
            mask = (assignments == dmbd_role)
            overlaps = (labels[mask] == gt_role).sum().item()
            role_counts[gt_role] = overlaps
        
        # Map to the most common ground truth role
        if role_counts:
            max_role = max(role_counts.items(), key=lambda x: x[1])[0]
            role_mapping[dmbd_role] = max_role
    
    # Map all assignments according to the role mapping
    mapped_assignments = torch.zeros_like(assignments)
    for dmbd_role, gt_role in role_mapping.items():
        mapped_assignments[assignments == dmbd_role] = gt_role
    
    # Calculate accuracy
    accuracy = (mapped_assignments == labels).float().mean().item()
    
    # Calculate accuracy per ground truth role
    per_role_accuracy = {}
    for gt_role in unique_labels:
        mask = (labels == gt_role)
        if mask.sum() > 0:
            per_role_accuracy[gt_role] = (mapped_assignments[mask] == gt_role).float().mean().item()
    
    # Store results
    results['accuracy'] = accuracy
    results['role_mapping'] = role_mapping
    results['unique_assignments'] = unique_assignments
    results['unique_labels'] = unique_labels
    results['per_role_accuracy'] = per_role_accuracy
    
    return results

def build_dmbd_model(feature_dim, reg_strength=0.01):
    """
    Build a DMBD model with regularization and dimension checks.
    
    Args:
        feature_dim: Dimension of input features
        reg_strength: Regularization strength
        
    Returns:
        DMBD model instance
    """
    try:
        from dmbd.dmbd import DMBD
        
        # Calculate appropriate dimensions based on feature_dim
        # Use sqrt for a reasonable balance between roles and hidden dims
        base_dim = max(2, int(np.sqrt(feature_dim)))
        role_dims = [base_dim] * 3  # One for each category (internal, blanket, external)
        hidden_dims = [base_dim] * 3  # Same for hidden states
        
        logger.info(f"Creating DMBD model with dimensions:")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Role dims: {role_dims}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        
        # Create model with calculated dimensions
        model = DMBD(
            obs_shape=(1, feature_dim),
            role_dims=role_dims,
            hidden_dims=hidden_dims,
            number_of_objects=1
        )
        
        # Apply regularization with dimension checks
        if hasattr(model, 'A') and hasattr(model.A, 'data'):
            A_shape = model.A.data.shape
            reg_matrix = torch.eye(A_shape[0]) * reg_strength
            model.A.data.add_(reg_matrix)
            logger.info(f"Applied regularization to A matrix of shape {A_shape}")
        
        if hasattr(model, 'C') and hasattr(model.C, 'data'):
            C_shape = model.C.data.shape
            reg_matrix = torch.eye(C_shape[0]) * reg_strength
            model.C.data.add_(reg_matrix)
            logger.info(f"Applied regularization to C matrix of shape {C_shape}")
        
        # Verify tensor dimensions
        _verify_model_dimensions(model, feature_dim)
        
        return model
    except Exception as e:
        logger.error(f"Error building DMBD model: {str(e)}")
        return None

def _verify_model_dimensions(model, feature_dim):
    """
    Verify that all model dimensions are consistent.
    
    Args:
        model: DMBD model instance
        feature_dim: Expected feature dimension
    
    Raises:
        ValueError: If dimensions are inconsistent
    """
    # Check observation shape
    if not hasattr(model, 'obs_shape') or model.obs_shape[-1] != feature_dim:
        raise ValueError(f"Model observation shape {getattr(model, 'obs_shape', None)} "
                        f"does not match feature dimension {feature_dim}")
    
    # Check role dimensions
    if not hasattr(model, 'role_dims') or len(model.role_dims) != 3:
        raise ValueError(f"Invalid role dimensions: {getattr(model, 'role_dims', None)}")
    
    # Check hidden dimensions
    if not hasattr(model, 'hidden_dims') or len(model.hidden_dims) != 3:
        raise ValueError(f"Invalid hidden dimensions: {getattr(model, 'hidden_dims', None)}")
    
    # Check matrix dimensions
    if hasattr(model, 'A') and hasattr(model.A, 'data'):
        A_shape = model.A.data.shape
        expected_A_dim = sum(model.hidden_dims)
        if A_shape[0] != expected_A_dim or A_shape[1] != expected_A_dim:
            raise ValueError(f"Invalid A matrix shape: {A_shape}, "
                           f"expected ({expected_A_dim}, {expected_A_dim})")
    
    if hasattr(model, 'C') and hasattr(model.C, 'data'):
        C_shape = model.C.data.shape
        if C_shape[1] != feature_dim:
            raise ValueError(f"Invalid C matrix shape: {C_shape}, "
                           f"expected second dimension to be {feature_dim}")

def run_dmbd_update(model, features, iterations=50, learning_rate=0.001, verbose=True):
    """
    Run DMBD update with dimension checks and error handling.
    
    Args:
        model: DMBD model instance
        features: Input features tensor
        iterations: Number of update iterations
        learning_rate: Learning rate for updates
        verbose: Whether to print progress
        
    Returns:
        tuple: (success, assignments)
    """
    try:
        # Verify input dimensions
        if len(features.shape) != 3:
            raise ValueError(f"Features should be 3D [time, batch, features], got shape {features.shape}")
        
        if features.shape[-1] != model.obs_shape[-1]:
            raise ValueError(f"Feature dimension {features.shape[-1]} does not match "
                           f"model observation shape {model.obs_shape[-1]}")
        
        # Log dimension information
        logger.info(f"Running DMBD update with dimensions:")
        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Model obs_shape: {model.obs_shape}")
        
        # Run update with progress tracking
        success = model.update(
            y=features,
            u=None,
            r=None,
            iters=iterations,
            lr=learning_rate,
            verbose=verbose
        )
        
        if success and hasattr(model, 'assignment') and callable(model.assignment):
            assignments = model.assignment()
            logger.info(f"Update successful, assignments shape: {assignments.shape}")
            return success, assignments
        else:
            logger.warning("Update completed but no assignments available")
            return success, None
            
    except Exception as e:
        logger.error(f"Error during DMBD update: {str(e)}")
        return False, None

def visualize_results(raw_data, labels, assignments, results, time_step, grid_size, output_dir):
    """
    Create visualizations of DMBD results.
    
    Args:
        raw_data: Raw input data tensor
        labels: Ground truth labels tensor
        assignments: DMBD role assignments tensor
        results: Dictionary with evaluation results
        time_step: Time step to visualize
        grid_size: Size of the grid
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log shapes for debugging
        logger.info(f"visualize_results - Assignments shape: {assignments.shape}")
        logger.info(f"visualize_results - Labels shape: {labels.shape}")
        logger.info(f"visualize_results - Raw data shape: {raw_data.shape}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot raw data
        raw_grid = raw_data[time_step, 0].reshape(grid_size, grid_size)
        axes[0].imshow(raw_grid, cmap='viridis')
        axes[0].set_title("Raw Data")
        
        # Plot ground truth labels
        gt_grid = labels[time_step].reshape(grid_size, grid_size)
        axes[1].imshow(gt_grid, cmap=ListedColormap(['red', 'green', 'blue']))
        axes[1].set_title("Ground Truth Labels")
        
        # Plot DMBD assignments
        if len(assignments.shape) == 3:
            # Handle [time_steps, channels, features]
            assign_grid = assignments[time_step, 0].reshape(grid_size, grid_size)
        elif assignments.shape[1] == 1:
            # Handle [time_steps, 1] (single assignment per time step)
            # Create a grid filled with the single assignment value
            assign_value = assignments[time_step, 0].item()
            assign_grid = torch.full((grid_size, grid_size), assign_value)
        else:
            # Handle [time_steps, grid_size**2]
            assign_grid = assignments[time_step].reshape(grid_size, grid_size)
            
        axes[2].imshow(assign_grid, cmap=ListedColormap(['red', 'green', 'blue']))
        axes[2].set_title(f"DMBD Assignments\nAccuracy: {results['accuracy']:.4f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "role_assignments.png"))
        plt.close()
        
        # Save a text summary of the results
        with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
            f.write(f"DMBD Evaluation Results\n")
            f.write(f"======================\n\n")
            f.write(f"Overall accuracy: {results['accuracy']:.4f}\n\n")
            f.write(f"Role mapping:\n")
            for dmbd_role, gt_role in results.get('role_mapping', {}).items():
                f.write(f"  DMBD role {dmbd_role} -> Ground truth role {gt_role}\n")
            f.write(f"\nPer-role accuracy:\n")
            for role, acc in results.get('per_role_accuracy', {}).items():
                f.write(f"  Role {role}: {acc:.4f}\n")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}") 