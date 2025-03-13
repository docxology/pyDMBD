#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for DMBD model to improve stability and handle tensor operations.

This module provides helper functions to address common issues in the DMBD model:
- Matrix regularization to prevent singular matrices
- Tensor dimension checking and reshaping
- Gradient handling and tracking
- Debugging utilities for tensor operations
"""

import torch
import numpy as np
import logging
import traceback
from typing import Tuple, Dict, List, Optional, Union, Any

# Configure logging
logger = logging.getLogger("dmbd_utils")

def set_up_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def regularize_matrix(matrix: torch.Tensor, strength: float = 1e-4) -> torch.Tensor:
    """
    Add regularization to a matrix to prevent singular matrices during inversion.
    
    Args:
        matrix: Input matrix tensor
        strength: Regularization strength (added to diagonal)
        
    Returns:
        Regularized matrix
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(matrix)}")
    
    if len(matrix.shape) < 2:
        raise ValueError(f"Expected matrix with at least 2 dimensions, got shape {matrix.shape}")
    
    # Get the last two dimensions (matrix dimensions)
    matrix_shape = matrix.shape[-2:]
    
    # Check if the matrix is square
    if matrix_shape[0] != matrix_shape[1]:
        raise ValueError(f"Cannot regularize non-square matrix with shape {matrix_shape}")
    
    # Create identity matrix with same device and dtype
    identity = torch.eye(
        matrix_shape[0], 
        device=matrix.device, 
        dtype=matrix.dtype
    )
    
    # Reshape identity if needed
    if len(matrix.shape) > 2:
        # Reshape identity to match batch dimensions
        for _ in range(len(matrix.shape) - 2):
            identity = identity.unsqueeze(0)
        
        # Expand identity to match matrix batch dimensions
        identity = identity.expand(*matrix.shape[:-2], matrix_shape[0], matrix_shape[1])
    
    # Add regularization
    regularized = matrix + strength * identity
    
    return regularized


def safe_matrix_inverse(matrix: torch.Tensor, 
                        reg_strength: float = 1e-4, 
                        svd_fallback: bool = True) -> torch.Tensor:
    """
    Safely compute matrix inverse with regularization and SVD fallback.
    
    Args:
        matrix: Input matrix tensor
        reg_strength: Regularization strength
        svd_fallback: Whether to use SVD-based pseudoinverse as fallback
        
    Returns:
        Inverted matrix
    """
    # First try with regularization
    regularized = regularize_matrix(matrix, strength=reg_strength)
    
    try:
        inverted = torch.linalg.inv(regularized)
        
        # Check for NaN or Inf values
        if torch.isnan(inverted).any() or torch.isinf(inverted).any():
            raise ValueError("Matrix inversion resulted in NaN or Inf values")
        
        return inverted
        
    except Exception as e:
        if not svd_fallback:
            raise e
        
        logger.warning(f"Standard matrix inversion failed: {str(e)}. Using SVD fallback.")
        
        # Fall back to SVD-based pseudoinverse
        try:
            # Handle batch dimensions
            orig_shape = matrix.shape
            matrix_dims = orig_shape[-2:]
            batch_dims = orig_shape[:-2]
            
            # Reshape to 2D
            matrix_2d = matrix.reshape(-1, matrix_dims[0], matrix_dims[1])
            
            # Apply SVD to each matrix in the batch
            inverted_list = []
            
            for i in range(matrix_2d.shape[0]):
                # Get individual matrix
                m = matrix_2d[i]
                
                # Compute SVD
                u, s, v = torch.svd(m)
                
                # Compute pseudoinverse of singular values with threshold
                threshold = reg_strength * torch.max(s)
                s_inv = torch.zeros_like(s)
                s_inv[s > threshold] = 1.0 / s[s > threshold]
                
                # Compute pseudoinverse
                pseudo_inv = v @ torch.diag(s_inv) @ u.t()
                inverted_list.append(pseudo_inv)
            
            # Stack and reshape back to original batch shape
            inverted = torch.stack(inverted_list).reshape(*batch_dims, matrix_dims[0], matrix_dims[1])
            
            return inverted
            
        except Exception as nested_e:
            logger.error(f"SVD fallback also failed: {str(nested_e)}")
            raise RuntimeError(f"Matrix inversion failed even with SVD fallback: {str(nested_e)}")


def check_tensor_dimensions(tensor_dict: Dict[str, torch.Tensor], 
                           expected_shapes: Dict[str, Tuple]) -> List[str]:
    """
    Check tensor dimensions against expected shapes.
    
    Args:
        tensor_dict: Dictionary of tensors
        expected_shapes: Dictionary of expected shapes
        
    Returns:
        List of dimension mismatch error messages (empty if all match)
    """
    errors = []
    
    for name, expected_shape in expected_shapes.items():
        if name not in tensor_dict:
            errors.append(f"Missing tensor: {name}")
            continue
        
        tensor = tensor_dict[name]
        actual_shape = tensor.shape
        
        # Check if shapes match
        if len(actual_shape) != len(expected_shape):
            errors.append(f"Tensor {name} has {len(actual_shape)} dimensions, expected {len(expected_shape)}")
            continue
        
        # Check each dimension
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            # If expected is None, any dimension is allowed
            if expected is not None and actual != expected:
                errors.append(f"Tensor {name} has shape {actual_shape}, expected {expected_shape} (mismatch at dim {i})")
                break
    
    return errors


def reshape_for_broadcasting(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Reshape tensors for broadcasting compatibility.
    
    Args:
        tensors: List of tensors to reshape
        
    Returns:
        List of reshaped tensors
    """
    if not tensors:
        return []
    
    # Determine maximum number of dimensions
    max_dims = max(len(t.shape) for t in tensors)
    
    # Reshape tensors
    reshaped = []
    for tensor in tensors:
        if len(tensor.shape) < max_dims:
            # Add dimensions to the left
            padding_dims = max_dims - len(tensor.shape)
            new_shape = (1,) * padding_dims + tensor.shape
            reshaped.append(tensor.reshape(new_shape))
        else:
            reshaped.append(tensor)
    
    return reshaped


def create_diagonal_tensor(batch_shape: Tuple, matrix_size: int, 
                          value: float = 1.0) -> torch.Tensor:
    """
    Create a batch of diagonal matrices.
    
    Args:
        batch_shape: Shape of batch dimensions
        matrix_size: Size of the diagonal matrix
        value: Value to put on the diagonal
        
    Returns:
        Tensor with shape (*batch_shape, matrix_size, matrix_size)
    """
    # Create identity matrix
    identity = torch.eye(matrix_size) * value
    
    # Add batch dimensions
    for _ in range(len(batch_shape)):
        identity = identity.unsqueeze(0)
    
    # Expand to batch shape
    batched = identity.expand(*batch_shape, matrix_size, matrix_size)
    
    return batched


def debug_tensor(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """
    Debug a tensor by collecting its properties.
    
    Args:
        tensor: Tensor to debug
        name: Name of the tensor for reference
        
    Returns:
        Dictionary of tensor properties
    """
    if tensor is None:
        return {"name": name, "status": "None"}
    
    try:
        properties = {
            "name": name,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "min": tensor.min().item() if tensor.numel() > 0 else None,
            "max": tensor.max().item() if tensor.numel() > 0 else None,
            "mean": tensor.mean().item() if tensor.numel() > 0 else None,
            "std": tensor.std().item() if tensor.numel() > 0 and tensor.dtype.is_floating_point else None,
            "has_nan": torch.isnan(tensor).any().item() if tensor.numel() > 0 else None,
            "has_inf": torch.isinf(tensor).any().item() if tensor.numel() > 0 else None,
            "numel": tensor.numel()
        }
        
        return properties
        
    except Exception as e:
        return {
            "name": name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def apply_model_regularization(model, strength=1e-4):
    """
    Apply regularization to matrices in a DMBD model.
    
    Args:
        model: DMBD model instance
        strength: Regularization strength
        
    Returns:
        Boolean indicating success
    """
    modified = False
    
    if hasattr(model, 'A') and hasattr(model.A, 'data'):
        # Add regularization to A prior precision
        if hasattr(model.A, 'prior_precision') and model.A.prior_precision is not None:
            try:
                hidden_dim = model.A.data.shape[0]
                model.A.prior_precision = model.A.prior_precision + strength * torch.eye(hidden_dim)
                modified = True
            except Exception as e:
                logger.warning(f"Failed to regularize A.prior_precision: {str(e)}")
    
    if hasattr(model, 'C') and hasattr(model.C, 'data'):
        # Add regularization to C prior precision
        if hasattr(model.C, 'prior_precision') and model.C.prior_precision is not None:
            try:
                hidden_dim = model.C.data.shape[-1]
                model.C.prior_precision = model.C.prior_precision + strength * torch.eye(hidden_dim)
                modified = True
            except Exception as e:
                logger.warning(f"Failed to regularize C.prior_precision: {str(e)}")
    
    # Check if model has covariance matrices
    for attr_name in ['Sigma', 'Gamma', 'Omega']:
        if hasattr(model, attr_name) and hasattr(getattr(model, attr_name), 'data'):
            attr = getattr(model, attr_name)
            try:
                # Get the matrix dimensions (last two dimensions)
                shape = attr.data.shape
                if len(shape) >= 2:
                    matrix_size = shape[-1]
                    if shape[-2] == matrix_size:  # Square matrix
                        attr.data = attr.data + strength * torch.eye(matrix_size)
                        modified = True
            except Exception as e:
                logger.warning(f"Failed to regularize {attr_name}: {str(e)}")
    
    return modified


def check_model_dimensions(model) -> Dict[str, Any]:
    """
    Check model dimensions for consistency.
    
    Args:
        model: DMBD model instance
        
    Returns:
        Dictionary with dimension information and potential issues
    """
    dimensions = {
        "model_attributes": {},
        "tensors": {},
        "issues": []
    }
    
    # Check key model attributes
    for attr_name in ['obs_shape', 'role_dims', 'hidden_dims', 'number_of_objects']:
        if hasattr(model, attr_name):
            dimensions["model_attributes"][attr_name] = getattr(model, attr_name)
    
    # Check key tensors
    for tensor_name in ['A', 'C', 'Sigma']:
        if hasattr(model, tensor_name):
            tensor = getattr(model, tensor_name)
            if hasattr(tensor, 'data'):
                dimensions["tensors"][tensor_name] = {
                    "shape": tensor.data.shape,
                    "dtype": str(tensor.data.dtype)
                }
    
    # Check for potential issues
    
    # 1. Check if A and C have compatible dimensions
    if 'A' in dimensions["tensors"] and 'C' in dimensions["tensors"]:
        a_shape = dimensions["tensors"]['A']["shape"]
        c_shape = dimensions["tensors"]['C']["shape"]
        
        # A should have shape (hidden_dim, hidden_dim)
        # C should have shape (obs_dim, hidden_dim) or (batch, obs_dim, hidden_dim)
        
        if len(a_shape) >= 2 and len(c_shape) >= 2:
            # Check if hidden dimensions match
            if a_shape[-1] != c_shape[-1]:
                dimensions["issues"].append(
                    f"Hidden dimension mismatch: A has {a_shape[-1]}, C has {c_shape[-1]}"
                )
    
    # 2. Check if obs_shape matches C matrix dimensions
    if 'obs_shape' in dimensions["model_attributes"] and 'C' in dimensions["tensors"]:
        obs_shape = dimensions["model_attributes"]["obs_shape"]
        c_shape = dimensions["tensors"]['C']["shape"]
        
        if len(c_shape) >= 2 and len(obs_shape) >= 1:
            # Check if observation dimension matches
            if obs_shape[-1] != c_shape[-2]:
                dimensions["issues"].append(
                    f"Observation dimension mismatch: obs_shape is {obs_shape}, but C has shape {c_shape}"
                )
    
    # 3. Check role dimensions
    if 'role_dims' in dimensions["model_attributes"]:
        role_dims = dimensions["model_attributes"]["role_dims"]
        
        if not isinstance(role_dims, list) or len(role_dims) != 3:
            dimensions["issues"].append(
                f"Expected role_dims to be a list of 3 integers, got {role_dims}"
            )
    
    return dimensions


def patch_model_for_stability(model, reg_strength=1e-4, modify_inversion=True):
    """
    Patch a DMBD model for numerical stability.
    
    Args:
        model: DMBD model instance
        reg_strength: Regularization strength
        modify_inversion: Whether to monkey patch matrix inversion
        
    Returns:
        Boolean indicating success
    """
    try:
        # Apply regularization to model matrices
        reg_success = apply_model_regularization(model, strength=reg_strength)
        
        # Monkey patch matrix inversion if requested
        if modify_inversion:
            # Store the original inversion function
            original_inv = torch.linalg.inv
            
            # Define a safer inversion function
            def safe_inv(x):
                return safe_matrix_inverse(x, reg_strength=reg_strength)
            
            # Monkey patch
            torch.linalg.inv = safe_inv
            
            # Store original function for restoration
            if not hasattr(model, '_original_inv'):
                model._original_inv = original_inv
        
        return reg_success
        
    except Exception as e:
        logger.error(f"Failed to patch model: {str(e)}")
        return False


def restore_model_patches(model):
    """
    Restore any monkey patches applied to the model.
    
    Args:
        model: DMBD model instance
        
    Returns:
        Boolean indicating success
    """
    try:
        # Restore original matrix inversion
        if hasattr(model, '_original_inv'):
            torch.linalg.inv = model._original_inv
            delattr(model, '_original_inv')
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Failed to restore model patches: {str(e)}")
        return False 