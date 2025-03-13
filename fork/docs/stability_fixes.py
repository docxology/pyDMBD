#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stability Fixes for Dynamic Markov Blanket Discovery (DMBD)

This module provides comprehensive fixes for numerical stability issues in DMBD.
It contains implementations for:
1. Robust matrix inversion
2. Tensor dimension consistency management
3. Resilient update mechanisms

These fixes can be applied directly to the DMBD core implementation or used
as monkey patches to improve stability without modifying the original code.
"""

import os
import sys
import torch
import logging
import numpy as np
import traceback
from functools import wraps
from typing import Dict, Tuple, List, Optional, Union, Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dmbd_stability")

# ----- MATRIX CONDITIONING AND INVERSION -----

def stable_inverse(tensor: torch.Tensor, eps_scale: float = 1e-6) -> torch.Tensor:
    """
    SVD-based pseudoinverse for robust matrix inversion of potentially singular matrices.
    
    Args:
        tensor: Input tensor to invert
        eps_scale: Scale for threshold relative to maximum singular value
        
    Returns:
        Stably inverted matrix
    """
    # Handle edge cases
    if tensor.numel() == 1:  # Singleton
        if tensor.item() == 0:
            return torch.tensor([[0.0]], device=tensor.device)
        return torch.tensor([[1.0 / tensor.item()]], device=tensor.device)
    
    # For 1D tensors, convert to diagonal matrix first
    if tensor.dim() == 1:
        tensor = torch.diag(tensor)
    
    # Use SVD for numerical stability
    try:
        U, S, V = torch.svd(tensor)
        
        # Filter small singular values for numerical stability
        eps = eps_scale * S.max()
        S_inv = torch.zeros_like(S)
        S_inv[S > eps] = 1.0 / S[S > eps]
        
        # Compute pseudo-inverse
        if tensor.shape[0] == tensor.shape[1]:  # Square matrix
            return V @ torch.diag(S_inv) @ U.t()
        else:  # Non-square matrix
            S_inv_mat = torch.zeros(V.shape[1], U.shape[1], device=tensor.device)
            for i in range(min(S.shape[0], V.shape[1], U.shape[1])):
                S_inv_mat[i, i] = S_inv[i]
            return V @ S_inv_mat @ U.t()
    except Exception as e:
        logger.error(f"SVD failed in stable_inverse: {str(e)}")
        # Even more robust fallback - use numpy with larger epsilon
        try:
            if tensor.dim() <= 2:  # Only works for matrices, not batched tensors
                cpu_tensor = tensor.cpu().numpy()
                # Use numpy's pseudoinverse with higher tolerance
                np_inv = np.linalg.pinv(cpu_tensor, rcond=eps_scale*10)
                return torch.tensor(np_inv, device=tensor.device)
            else:
                # For batched tensors, add regularization and try again
                eye_size = tensor.shape[-1]
                batch_shape = tensor.shape[:-2]
                eye = torch.eye(eye_size, device=tensor.device)
                for _ in range(len(batch_shape)):
                    eye = eye.unsqueeze(0)
                eye = eye.expand(*batch_shape, eye_size, eye_size)
                regularized = tensor + (eps_scale * 100) * eye
                return torch.linalg.inv(regularized)
        except Exception as e2:
            logger.error(f"Both SVD and numpy fallback failed: {str(e2)}")
            # Last resort - return identity matrix
            eye_size = tensor.shape[-1]
            batch_shape = tensor.shape[:-2]
            eye = torch.eye(eye_size, device=tensor.device)
            for _ in range(len(batch_shape)):
                eye = eye.unsqueeze(0)
            eye = eye.expand(*batch_shape, eye_size, eye_size)
            return eye


def ensure_positive_definite(tensor: torch.Tensor, 
                            min_eigenval: float = 1e-6) -> torch.Tensor:
    """
    Ensure a matrix is positive definite by adding small values to diagonal if needed.
    
    Args:
        tensor: Input tensor to make positive definite
        min_eigenval: Minimum eigenvalue to ensure
        
    Returns:
        Positive definite version of input tensor
    """
    if tensor.dim() < 2 or tensor.shape[-1] != tensor.shape[-2]:
        # Not a square matrix in the last two dimensions
        return tensor
    
    # For small matrices, directly compute eigenvalues
    if tensor.shape[-1] <= 10:
        try:
            # Get batch shape
            batch_shape = tensor.shape[:-2]
            n = tensor.shape[-1]
            
            # Process each batch element
            result = tensor.clone()
            
            # Use symmetric eigendecomposition for numerical stability
            symmetric_tensor = 0.5 * (tensor + tensor.transpose(-2, -1))
            
            if len(batch_shape) == 0:
                # Non-batched case
                eigvals, eigvecs = torch.linalg.eigh(symmetric_tensor)
                # Set minimum eigenvalue
                eigvals = torch.clamp(eigvals, min=min_eigenval)
                # Reconstruct matrix
                result = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
            else:
                # Batched case
                flat_batch = torch.prod(torch.tensor(batch_shape)).item()
                reshaped = symmetric_tensor.reshape(flat_batch, n, n)
                
                for i in range(flat_batch):
                    eigvals, eigvecs = torch.linalg.eigh(reshaped[i])
                    # Set minimum eigenvalue
                    eigvals = torch.clamp(eigvals, min=min_eigenval)
                    # Reconstruct matrix
                    reshaped[i] = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
                
                result = reshaped.reshape(*batch_shape, n, n)
            
            return result
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {str(e)}. Using diagonal regularization.")
    
    # Fallback: add to diagonal (faster for large matrices)
    eye_size = tensor.shape[-1]
    batch_shape = tensor.shape[:-2]
    eye = torch.eye(eye_size, device=tensor.device)
    
    # Handle batch dimensions
    for _ in range(len(batch_shape)):
        eye = eye.unsqueeze(0)
    eye = eye.expand(*batch_shape, eye_size, eye_size)
    
    # Add regularization based on the magnitude of the tensor
    reg_strength = min_eigenval * (1.0 + tensor.abs().max())
    return tensor + reg_strength * eye


def robust_matrix_ops(dmbd_instance: Any) -> None:
    """
    Apply robust matrix operations to a DMBD instance.
    
    This patches the instance to use stability-enhanced matrix operations
    for all internal calculations.
    
    Args:
        dmbd_instance: The DMBD instance to patch
    """
    # Store original linalg.inv for reference
    original_inv = torch.linalg.inv
    
    # Replace torch.linalg.inv with stable version globally for this module
    def patched_inv(input, *args, **kwargs):
        try:
            return stable_inverse(input)
        except Exception as e:
            logger.error(f"Error in patched_inv: {str(e)}")
            return original_inv(input, *args, **kwargs)
    
    # Monkey patch the torch.linalg.inv function within the DMBD module
    torch.linalg.inv = patched_inv
    
    # Add utility methods to the instance
    dmbd_instance.stable_inverse = stable_inverse
    dmbd_instance.ensure_positive_definite = ensure_positive_definite
    
    logger.info("Applied robust matrix operations to DMBD instance")


# ----- TENSOR DIMENSION CONSISTENCY -----

def ensure_batch_consistency(tensors: Dict[str, torch.Tensor], 
                            batch_dim: int) -> Dict[str, torch.Tensor]:
    """
    Ensures all tensors have consistent batch dimensions.
    
    Args:
        tensors: Dictionary of tensor names to tensors
        batch_dim: Target batch dimension size
        
    Returns:
        Dictionary of tensor names to batch-consistent tensors
    """
    result = {}
    for name, tensor in tensors.items():
        if tensor is None or not isinstance(tensor, torch.Tensor):
            result[name] = tensor
            continue
        
        if len(tensor.shape) > 0 and tensor.shape[0] != batch_dim:
            if tensor.shape[0] == 1:  # Singleton batch
                # Broadcast singleton batch to match
                result[name] = tensor.expand(batch_dim, *tensor.shape[1:])
                logger.debug(f"Broadcasting tensor '{name}' from shape {tensor.shape} to {result[name].shape}")
            elif tensor.shape[0] > batch_dim:  # Too large
                # Slice to match
                result[name] = tensor[:batch_dim]
                logger.debug(f"Slicing tensor '{name}' from shape {tensor.shape} to {result[name].shape}")
            else:  # Too small
                # Pad by repeating last element
                padding = tensor[-1:].expand(batch_dim - tensor.shape[0], *tensor.shape[1:])
                result[name] = torch.cat([tensor, padding], dim=0)
                logger.debug(f"Padding tensor '{name}' from shape {tensor.shape} to {result[name].shape}")
        else:
            result[name] = tensor
    
    return result


def validate_tensor_dims(tensor: torch.Tensor, 
                        expected_shape: Tuple[int, ...], 
                        name: str = "tensor",
                        fix_if_possible: bool = True) -> torch.Tensor:
    """
    Validates and potentially fixes tensor dimensions.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (use -1 for dimensions that can be any size)
        name: Name of tensor for logging
        fix_if_possible: Whether to try fixing the shape if it doesn't match
        
    Returns:
        Tensor with validated (and potentially fixed) shape
    """
    if tensor is None:
        return None
    
    # Check if shape matches expected shape, ignoring dimensions with -1
    shape_ok = len(tensor.shape) == len(expected_shape)
    if shape_ok:
        for actual, expected in zip(tensor.shape, expected_shape):
            if expected != -1 and actual != expected:
                shape_ok = False
                break
    
    if not shape_ok and fix_if_possible:
        try:
            # Try to reshape or broadcast to match expected shape
            new_shape = list(expected_shape)
            
            # Replace -1 entries with actual dimensions
            for i, dim in enumerate(new_shape):
                if dim == -1 and i < len(tensor.shape):
                    new_shape[i] = tensor.shape[i]
            
            # If dimensions with same total elements, reshape
            if tensor.numel() == np.prod([d for d in new_shape if d > 0]):
                fixed_tensor = tensor.reshape(new_shape)
                logger.info(f"Reshaped {name} from {tensor.shape} to {fixed_tensor.shape}")
                return fixed_tensor
            
            # If singleton dimension can be broadcast
            could_broadcast = True
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected != -1 and actual != expected and actual != 1:
                    could_broadcast = False
                    break
            
            if could_broadcast:
                # Create broadcast shape, preserving actual dimensions where expected is -1
                broadcast_shape = []
                for i, expected in enumerate(expected_shape):
                    if expected == -1 and i < len(tensor.shape):
                        broadcast_shape.append(tensor.shape[i])
                    else:
                        broadcast_shape.append(expected)
                
                fixed_tensor = tensor.expand(broadcast_shape)
                logger.info(f"Broadcast {name} from {tensor.shape} to {fixed_tensor.shape}")
                return fixed_tensor
        except Exception as e:
            logger.warning(f"Failed to fix shape of {name}: {str(e)}")
    
    if not shape_ok:
        logger.warning(f"Tensor {name} has shape {tensor.shape}, expected {expected_shape}")
    
    return tensor


def dimension_aware_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with automatic dimension adjustment.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result of matrix multiplication
    """
    try:
        return torch.matmul(a, b)
    except RuntimeError as e:
        if "mat1 and mat2 shapes cannot be multiplied" in str(e):
            # Try to reshape for compatibility
            if a.dim() >= 2 and b.dim() >= 2:
                # Get the inner dimensions that should match
                a_inner = a.shape[-1]
                b_inner = b.shape[-2]
                
                if a_inner != b_inner:
                    # Possible broadcasting case
                    if a_inner == 1 and b_inner > 1:
                        # Broadcast a's last dimension
                        a_reshaped = a.expand(*a.shape[:-1], b_inner)
                        return torch.matmul(a_reshaped, b)
                    elif b_inner == 1 and a_inner > 1:
                        # Broadcast b's second-to-last dimension
                        b_reshaped = b.expand(*b.shape[:-2], a_inner, b.shape[-1])
                        return torch.matmul(a, b_reshaped)
            
            # If we got here, we couldn't fix it
            logger.error(f"Failed to adjust dimensions for matmul: {a.shape} @ {b.shape}")
        raise


def fix_dimension_handling(dmbd_instance: Any) -> None:
    """
    Apply dimension consistency fixes to a DMBD instance.
    
    Args:
        dmbd_instance: The DMBD instance to patch
    """
    # Add utility methods to the instance
    dmbd_instance.ensure_batch_consistency = ensure_batch_consistency
    dmbd_instance.validate_tensor_dims = validate_tensor_dims
    dmbd_instance.dimension_aware_matmul = dimension_aware_matmul
    
    # Store original torch.matmul for reference
    original_matmul = torch.matmul
    
    # Replace torch.matmul with dimension-aware version
    torch.matmul = dimension_aware_matmul
    
    # Patch key methods with consistency checks if they exist
    if hasattr(dmbd_instance, 'update_obs_parms'):
        original_update_obs = dmbd_instance.update_obs_parms
        
        @wraps(original_update_obs)
        def consistent_update_obs(self, *args, **kwargs):
            try:
                # First check if latent parameters are initialized
                if not hasattr(self, 'mu_r') or self.mu_r is None:
                    logger.warning("DMBD update_obs_parms: latent not initialized")
                    return False
                
                # Ensure dimensions are consistent before updating
                batch_dim = self.Y.shape[0]
                
                # Collect all relevant tensors
                tensors_to_check = {'mu_r': self.mu_r}
                if hasattr(self, 'Y_for_r'):
                    tensors_to_check['Y_for_r'] = self.Y_for_r
                
                # Apply batch consistency
                consistent = ensure_batch_consistency(tensors_to_check, batch_dim)
                for name, tensor in consistent.items():
                    setattr(self, name, tensor)
                
                # Now call the original update
                return original_update_obs(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in consistent_update_obs: {str(e)}")
                logger.error(traceback.format_exc())
                return False
        
        # Replace the method
        dmbd_instance.update_obs_parms = consistent_update_obs.__get__(dmbd_instance, type(dmbd_instance))
    
    if hasattr(dmbd_instance, 'update_latent_parms'):
        original_update_latent = dmbd_instance.update_latent_parms
        
        @wraps(original_update_latent)
        def consistent_update_latent(self, *args, **kwargs):
            try:
                # Ensure dimensions are consistent before updating
                if not hasattr(self, 'Y') or self.Y is None:
                    logger.warning("DMBD update_latent_parms: observations not available")
                    return False
                
                batch_dim = self.Y.shape[0]
                
                # Check and fix invSigma matrices dimensions
                tensors_to_check = {}
                for attr_name in dir(self):
                    if attr_name.startswith('invSigma_') and isinstance(getattr(self, attr_name), torch.Tensor):
                        tensors_to_check[attr_name] = getattr(self, attr_name)
                
                # Apply batch consistency
                consistent = ensure_batch_consistency(tensors_to_check, batch_dim)
                for name, tensor in consistent.items():
                    setattr(self, name, tensor)
                
                # Now call the original update
                return original_update_latent(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in consistent_update_latent: {str(e)}")
                logger.error(traceback.format_exc())
                return False
        
        # Replace the method
        dmbd_instance.update_latent_parms = consistent_update_latent.__get__(dmbd_instance, type(dmbd_instance))
    
    logger.info("Applied dimension consistency fixes to DMBD instance")


# ----- ROBUST UPDATE MECHANISMS -----

def exception_handler(func: Callable) -> Callable:
    """
    Decorator for handling exceptions in DMBD operations.
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    return wrapper


def adaptive_regularized_update(dmbd_instance: Any,
                               features: torch.Tensor,
                               inputs: Optional[torch.Tensor] = None,
                               roles: Optional[torch.Tensor] = None,
                               iterations: int = 100,
                               learning_rate: float = 0.001,
                               initial_reg: float = 1e-4,
                               max_attempts: int = 3,
                               verbose: bool = True) -> bool:
    """
    Update DMBD with adaptive regularization and multiple attempts.
    
    Args:
        dmbd_instance: DMBD instance to update
        features: Feature tensor (observations)
        inputs: Input tensor (if any)
        roles: Role tensor (if any)
        iterations: Number of update iterations
        learning_rate: Learning rate for updates
        initial_reg: Initial regularization strength
        max_attempts: Maximum number of update attempts
        verbose: Whether to print progress
        
    Returns:
        Success flag
    """
    success = False
    attempts = 0
    reg_strength = initial_reg
    
    # Save original update method
    original_update = dmbd_instance.update
    
    while not success and attempts < max_attempts:
        try:
            if verbose:
                logger.info(f"Update attempt {attempts+1}/{max_attempts} with reg={reg_strength:.2e}")
            
            # Apply regularization
            if hasattr(dmbd_instance, 'regularization_strength'):
                # Direct attribute setting if available
                old_reg = dmbd_instance.regularization_strength
                dmbd_instance.regularization_strength = reg_strength
            
            # Try the update
            success = original_update(
                y=features,
                u=inputs,
                r=roles,
                iters=iterations,
                lr=learning_rate,
                verbose=verbose
            )
            
            # If succeeded, return
            if success:
                if verbose:
                    logger.info(f"Update succeeded after {attempts+1} attempts")
                return True
            
            # Restore original regularization
            if hasattr(dmbd_instance, 'regularization_strength'):
                dmbd_instance.regularization_strength = old_reg
            
        except Exception as e:
            logger.warning(f"Update failed with reg={reg_strength:.2e}: {str(e)}")
        
        # Increase regularization for next attempt
        reg_strength *= 10
        attempts += 1
    
    if not success and verbose:
        logger.warning(f"Update failed after {max_attempts} attempts")
    
    return success


def add_robust_updates(dmbd_instance: Any) -> None:
    """
    Add robust update mechanisms to a DMBD instance.
    
    Args:
        dmbd_instance: The DMBD instance to enhance
    """
    # Add utility methods to the instance
    dmbd_instance.exception_handler = exception_handler
    dmbd_instance.adaptive_regularized_update = lambda *args, **kwargs: adaptive_regularized_update(dmbd_instance, *args, **kwargs)
    
    # Wrap the original update method with the adaptive version
    original_update = dmbd_instance.update
    
    @wraps(original_update)
    def robust_update(self, y, u=None, r=None, iters=100, lr=0.001, verbose=True, max_attempts=3, initial_reg=1e-4):
        if max_attempts <= 1:
            # Use original update directly
            return original_update(y=y, u=u, r=r, iters=iters, lr=lr, verbose=verbose)
        else:
            # Use adaptive update
            return adaptive_regularized_update(
                dmbd_instance=self,
                features=y,
                inputs=u,
                roles=r,
                iterations=iters,
                learning_rate=lr,
                initial_reg=initial_reg,
                max_attempts=max_attempts,
                verbose=verbose
            )
    
    # Replace the update method
    dmbd_instance.update = robust_update.__get__(dmbd_instance, type(dmbd_instance))
    
    logger.info("Added robust update mechanisms to DMBD instance")


# ----- COMPLETE STABILITY ENHANCEMENT -----

def apply_all_stability_fixes(dmbd_instance: Any) -> None:
    """
    Apply all stability fixes to a DMBD instance.
    
    This is the main entry point for applying the complete set of
    stability enhancements to a DMBD instance.
    
    Args:
        dmbd_instance: The DMBD instance to enhance
    """
    logger.info("Applying all stability fixes to DMBD instance")
    
    # Apply matrix conditioning fixes
    robust_matrix_ops(dmbd_instance)
    
    # Apply dimension consistency fixes
    fix_dimension_handling(dmbd_instance)
    
    # Apply robust update mechanisms
    add_robust_updates(dmbd_instance)
    
    # Add convenience method to show this instance has been enhanced
    dmbd_instance.has_stability_fixes = True
    
    logger.info("Successfully applied all stability fixes")


# ----- MAIN USE EXAMPLE -----

def main():
    """Example usage of stability fixes."""
    try:
        # Try to import DMBD
        from dmbd.dmbd import DMBD
        
        # Create a basic DMBD model
        model = DMBD(
            obs_shape=(1, 9),
            role_dims=[3, 3, 3],
            hidden_dims=[3, 3, 3],
            number_of_objects=1
        )
        
        # Apply all stability fixes
        apply_all_stability_fixes(model)
        
        # Now the model has enhanced stability
        print("DMBD model enhanced with stability fixes")
        print("Available stability methods:")
        for method in ["stable_inverse", "ensure_positive_definite", 
                       "ensure_batch_consistency", "adaptive_regularized_update"]:
            if hasattr(model, method):
                print(f"- {method}")
        
    except ImportError:
        print("DMBD module not found. This module provides stability fixes for DMBD.")
        print("Install DMBD or ensure it's in your Python path to use these fixes.")


if __name__ == "__main__":
    main() 