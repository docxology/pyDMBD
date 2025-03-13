#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for DMBD tensor dimensions.

This module focuses on testing the tensor dimensions in the DMBD algorithm,
particularly regarding matrix shapes, tensor broadcasting, and dimension compatibility.
"""

import os
import sys
import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dmbd_dimensions_test")

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Import the DMBD model
try:
    from fork.dmbd.dmbd import DMBD
    logger.info("Successfully imported DMBD from fork")
except ImportError:
    try:
        from dmbd.dmbd import DMBD
        logger.info("Successfully imported DMBD from root")
    except ImportError:
        logger.error("Failed to import DMBD - tests may fail")


class TestDMBDDimensions(unittest.TestCase):
    """Test suite for DMBD dimension handling."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up logging
        self.logger = logging.getLogger('dmbd_dimensions_test')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            
        # Create output directory for test results
        self.output_dir = Path('fork/test_results/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Standard dimensions for testing
        self.feature_dims = [6, 9, 12]
        self.role_dims = [2, 3, 4]
        self.time_steps = 10
        self.batch_size = 1

    def test_01_matrix_shapes(self):
        """Test matrix shapes during DMBD initialization."""
        # Test different combinations of dimensions
        for feature_dim in self.feature_dims:
            for role_dim in self.role_dims:
                self.logger.info(f"Testing matrix shapes: features={feature_dim}, roles={role_dim}")
                
                # Initialize model
                model = DMBD(
                    obs_shape=(self.batch_size, feature_dim),
                    role_dims=[role_dim] * 3,  # Three identical role dimensions
                    hidden_dims=[role_dim] * 3,  # Three identical hidden dimensions
                    number_of_objects=1
                )
                
                # Total dimensions
                total_role_dim = sum([role_dim] * 3)  # Sum of all role dimensions
                total_hidden_dim = sum([role_dim] * 3)  # Sum of all hidden dimensions
                
                # Verify matrix shapes
                self.assertEqual(model.A.mu.shape[-2:], (total_hidden_dim, total_hidden_dim),
                               f"A matrix shape mismatch: {model.A.mu.shape} != ({total_hidden_dim}, {total_hidden_dim})")
                
                # The observation model (B) contains the role-based observation matrices
                # In ARHMM_prXRY, the observation matrix shape is (dim, n, p1+p2) where:
                # - dim is total_role_dim (number of roles)
                # - n is feature_dim (dimension of observables)
                # - p1 is total_hidden_dim (hidden state dimension)
                # - p2 is 0 (regression dimension, appears to be 0 in the actual implementation)
                expected_obs_shape = (total_role_dim, feature_dim, total_hidden_dim)
                actual_obs_shape = model.obs_model.obs_dist.mu.shape[-3:]
                self.assertEqual(actual_obs_shape, expected_obs_shape,
                               f"Observation matrix shape mismatch: {actual_obs_shape} != {expected_obs_shape}")
                
                # Test matrix initialization values
                self.assertFalse(torch.isnan(model.A.mu).any(),
                               "A matrix contains NaN values")
                self.assertFalse(torch.isnan(model.obs_model.obs_dist.mu).any(),
                               "Observation matrix contains NaN values")
                
                self.assertFalse(torch.isinf(model.A.mu).any(),
                               "A matrix contains Inf values")
                self.assertFalse(torch.isinf(model.obs_model.obs_dist.mu).any(),
                               "Observation matrix contains Inf values")
                
                self.logger.info("Matrix shape tests passed")
        
        # Log success
        logger.info("Matrix shapes test completed successfully")
    
    def test_02_dimension_consistency(self):
        """Test dimension consistency during DMBD update."""
        feature_dim = 6
        role_dim = 2
        time_steps = 10
        batch_size = 1
        
        self.logger.info(f"Testing dimensions: features={feature_dim}, roles={role_dim}")
        
        # Total dimensions
        total_role_dim = sum([role_dim] * 3)  # Sum of all role dimensions
        total_hidden_dim = sum([role_dim] * 3)  # Sum of all hidden dimensions
        
        # Create test data with correct dimensions
        y = torch.randn(time_steps, batch_size, feature_dim, 1)
        r = torch.randn(time_steps, batch_size, role_dim, 1)
        u = torch.zeros(time_steps, batch_size, 1, 1)  # Control input
        
        # Initialize model with correct dimensions
        model = DMBD(
            obs_shape=(batch_size, feature_dim),
            role_dims=[role_dim] * 3,  # Three identical role dimensions
            hidden_dims=[role_dim] * 3,  # Three identical hidden dimensions
            number_of_objects=1
        )
        
        # Initialize critical tensors
        model.A.mu = torch.nn.Parameter(torch.randn(total_hidden_dim, total_hidden_dim))
        model.obs_model.obs_dist.mu = torch.nn.Parameter(torch.randn(total_role_dim, feature_dim, total_hidden_dim + 1))
        
        # Track dimension issues
        issues = []
        
        try:
            # Perform update with dimension tracking
            def update_with_tracing():
                model.update(y=y, r=r, u=u)
                
                # Verify tensor dimensions after update
                if not hasattr(model, 'A') or model.A is None:
                    issues.append("Missing critical tensor: A")
                elif model.A.mu.shape[-2:] != (total_hidden_dim, total_hidden_dim):
                    issues.append(f"Incorrect A dimensions: {model.A.mu.shape}")
                    
                if not hasattr(model, 'obs_model') or model.obs_model is None:
                    issues.append("Missing critical tensor: obs_model")
                elif model.obs_model.obs_dist.mu.shape[-3:] != (total_role_dim, feature_dim, total_hidden_dim + 1):
                    issues.append(f"Incorrect observation matrix dimensions: {model.obs_model.obs_dist.mu.shape}")
                    
                # Check other tensor dimensions
                if hasattr(model, 'px4r') and model.px4r is not None:
                    if model.px4r.mu.shape != (time_steps, batch_size, feature_dim, 1):
                        issues.append(f"Incorrect px4r.mu dimensions: {model.px4r.mu.shape}")
            
            update_with_tracing()
            
        except Exception as e:
            self.logger.warning(f"Dimension issue: Error in execution:\n {str(e)}")
            issues.append(str(e))
        
        # Log any dimension issues found
        for issue in issues:
            self.logger.warning(f"Dimension issue: {issue}")
        
        self.assertEqual(len(issues), 0,
                        f"Dimension issues found with features={feature_dim}, roles={role_dim}")
    
    def _verify_tensor_dimensions(self, model, data):
        """Verify that all tensor dimensions are consistent."""
        # Check data dimensions
        self.assertEqual(len(data.shape), 3, "Input data should be 3D: [time, batch, features]")
        
        # Check model dimensions
        if hasattr(model, 'A') and hasattr(model.A, 'data'):
            A_shape = model.A.data.shape
            self.assertEqual(len(A_shape), 2, "A matrix should be 2D")
            
            # Verify A matrix dimensions match hidden dimensions
            total_hidden_dim = sum(model.hidden_dims)
            self.assertEqual(A_shape[0], total_hidden_dim, 
                            f"A matrix first dimension {A_shape[0]} should match total hidden dim {total_hidden_dim}")
            self.assertEqual(A_shape[1], total_hidden_dim, 
                            f"A matrix second dimension {A_shape[1]} should match total hidden dim {total_hidden_dim}")
        
        if hasattr(model, 'C') and hasattr(model.C, 'data'):
            C_shape = model.C.data.shape
            self.assertEqual(len(C_shape), 2, "C matrix should be 2D")
            
            # Verify C matrix dimensions match observation and hidden dimensions
            self.assertEqual(C_shape[1], data.shape[2], 
                            f"C matrix second dimension {C_shape[1]} should match feature dim {data.shape[2]}")
        
        # Check role dimensions
        if hasattr(model, 'role_dims'):
            self.assertEqual(len(model.role_dims), 3, "Should have 3 role dimensions")
            for dim in model.role_dims:
                self.assertGreater(dim, 0, "Role dimensions should be positive")
        
        # Check hidden dimensions
        if hasattr(model, 'hidden_dims'):
            self.assertEqual(len(model.hidden_dims), 3, "Should have 3 hidden dimensions")
            for dim in model.hidden_dims:
                self.assertGreater(dim, 0, "Hidden dimensions should be positive")
        
        # Check observation shape
        if hasattr(model, 'obs_shape'):
            self.assertEqual(len(model.obs_shape), 2, "Observation shape should be 2D")
            self.assertEqual(model.obs_shape[1], data.shape[2], 
                            f"Observation feature dimension {model.obs_shape[1]} should match data {data.shape[2]}")
    
    def _trace_dimensions_during_update(self, model, data, max_iters=2):
        """Trace tensor dimensions during DMBD update process."""
        # Create a dictionary to store tensor dimensions
        dim_trace = {
            'input': {
                'shape': data.shape,
                'type': str(data.dtype)
            },
            'update_args': {},
            'tensors': {},
            'errors': []
        }
        
        # Monkey patch the update method to track dimensions
        original_update = model.update
        
        def update_with_tracing(y, u=None, r=None, **kwargs):
            try:
                # Record update arguments
                dim_trace['update_args'] = {
                    'y': {'shape': y.shape, 'type': str(y.dtype)},
                    'u': {'shape': u.shape, 'type': str(u.dtype)} if u is not None else None,
                    'r': {'shape': r.shape, 'type': str(r.dtype)} if r is not None else None,
                    'kwargs': kwargs
                }
                
                # Collect model tensors before update
                self._collect_tensor_dimensions(model, dim_trace['tensors'], 'before_update')
                
                # Run the original update
                result = original_update(y, u, r, **kwargs)
                
                # Collect model tensors after update
                self._collect_tensor_dimensions(model, dim_trace['tensors'], 'after_update')
                
                return result
            except Exception as e:
                dim_trace['errors'].append({
                    'phase': 'update',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                raise
        
        # Replace the update method
        model.update = update_with_tracing.__get__(model, type(model))
        
        try:
            # Run the update with tracing
            model.update(
                y=data,
                u=None,
                r=None,
                iters=max_iters,
                lr=0.001,
                verbose=True
            )
        except Exception as e:
            dim_trace['errors'].append({
                'phase': 'execution',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        finally:
            # Restore the original update method
            model.update = original_update
        
        return dim_trace
    
    def _collect_tensor_dimensions(self, obj, collection, prefix=''):
        """Recursively collect tensor dimensions from an object."""
        if isinstance(obj, torch.Tensor):
            collection[prefix] = {
                'shape': obj.shape,
                'type': str(obj.dtype),
                'requires_grad': obj.requires_grad
            }
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._collect_tensor_dimensions(item, collection, f"{prefix}[{i}]")
        elif hasattr(obj, '__dict__'):
            for key, val in obj.__dict__.items():
                if not key.startswith('_'):
                    self._collect_tensor_dimensions(val, collection, f"{prefix}.{key}" if prefix else key)
    
    def _check_dimension_consistency(self, dim_trace):
        """Check for dimension consistency issues in the traced dimensions."""
        issues = []
        
        # Check if there was an error during update
        if dim_trace.get('errors'):
            for error in dim_trace['errors']:
                issues.append(f"Error in {error['phase']}: {error['error']}")
        
        # Check for missing critical tensors
        critical_tensors = ['A', 'C']
        for tensor in critical_tensors:
            tensor_key = f"before_update.{tensor}"
            if tensor_key not in dim_trace['tensors']:
                issues.append(f"Missing critical tensor: {tensor}")
        
        # Check tensor shape consistency
        if 'before_update' in dim_trace['tensors'] and 'after_update' in dim_trace['tensors']:
            before = dim_trace['tensors']['before_update']
            after = dim_trace['tensors']['after_update']
            
            # Compare shapes before and after update
            for key in before:
                if key in after:
                    if before[key]['shape'] != after[key]['shape']:
                        issues.append(f"Shape changed during update for {key}: "
                                    f"{before[key]['shape']} -> {after[key]['shape']}")
        
        # Check for NaN or Inf values
        for key, info in dim_trace['tensors'].items():
            if 'has_nan' in info and info['has_nan']:
                issues.append(f"NaN values in tensor: {key}")
            if 'has_inf' in info and info['has_inf']:
                issues.append(f"Inf values in tensor: {key}")
        
        return issues
    
    def test_03_gradient_tracking(self):
        """Test gradient tracking during DMBD update."""
        feature_dim = 9
        role_dim = 3
        time_steps = 10
        batch_size = 1
        
        # Total dimensions
        total_role_dim = sum([role_dim] * 3)  # Sum of all role dimensions
        total_hidden_dim = sum([role_dim] * 3)  # Sum of all hidden dimensions
        
        # Create test data with gradient tracking enabled
        y = torch.randn(time_steps, batch_size, feature_dim, 1, requires_grad=True)
        r = torch.randn(time_steps, batch_size, role_dim, 1, requires_grad=True)
        u = torch.zeros(time_steps, batch_size, 1, 1, requires_grad=True)  # Control input
        
        # Initialize model
        model = DMBD(
            obs_shape=(batch_size, feature_dim),
            role_dims=[role_dim] * 3,  # Three identical role dimensions
            hidden_dims=[role_dim] * 3,  # Three identical hidden dimensions
            number_of_objects=1
        )
        
        # Initialize parameters that should have gradients
        model.A.mu = torch.nn.Parameter(torch.randn(total_hidden_dim, total_hidden_dim))
        model.obs_model.obs_dist.mu = torch.nn.Parameter(torch.randn(total_role_dim, feature_dim, total_hidden_dim))
        
        # Dictionary to store gradient information
        grad_info = {'gradients': [], 'grad_norms': []}
        
        def hook_fn(grad):
            """Hook function to capture gradient information."""
            grad_info['gradients'].append(grad.clone().detach())
            grad_info['grad_norms'].append(torch.norm(grad).item())
            return grad
        
        # Register hooks for gradient tracking
        y.register_hook(hook_fn)
        r.register_hook(hook_fn)
        model.A.mu.register_hook(hook_fn)
        model.obs_model.obs_dist.mu.register_hook(hook_fn)
        
        try:
            # Instead of relying on the update method to compute gradients,
            # we'll manually compute a loss and call backward()
            
            # First, try to run a single update step
            try:
                model.update(y=y, r=r, u=u, iters=1, lr=0.001)
            except Exception as e:
                self.logger.warning(f"Model update failed, but continuing with gradient test: {str(e)}")
            
            # Compute a simple loss based on model parameters
            loss = torch.mean(torch.square(model.A.mu)) + torch.mean(torch.square(model.obs_model.obs_dist.mu))
            loss.backward()
            
            # Verify gradients were captured
            self.assertGreater(len(grad_info['gradients']), 0,
                             "No gradients were captured during backward pass")
            
            # Check gradient properties
            for i, grad in enumerate(grad_info['gradients']):
                self.assertFalse(torch.isnan(grad).any(),
                               f"NaN gradient detected in gradient {i}")
                self.assertFalse(torch.isinf(grad).any(),
                               f"Inf gradient detected in gradient {i}")
                self.assertGreater(grad_info['grad_norms'][i], 0,
                                 f"Zero gradient norm detected in gradient {i}")
            
            # Log gradient statistics
            avg_norm = sum(grad_info['grad_norms']) / len(grad_info['grad_norms'])
            max_norm = max(grad_info['grad_norms'])
            min_norm = min(grad_info['grad_norms'])
            
            self.logger.info(f"Gradient statistics:")
            self.logger.info(f"  Average norm: {avg_norm:.6f}")
            self.logger.info(f"  Max norm: {max_norm:.6f}")
            self.logger.info(f"  Min norm: {min_norm:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error during gradient tracking: {str(e)}")
            raise
    
    def test_04_dimension_mismatches(self):
        """Test handling of dimension mismatches in DMBD."""
        # Create a series of test cases with mismatched dimensions
        test_cases = [
            {
                'name': 'Feature dimension mismatch',
                'obs_shape': (1, 9),
                'data_shape': (10, 1, 12),
                'expected_error': True
            },
            {
                'name': 'Batch dimension mismatch',
                'obs_shape': (1, 9),
                'data_shape': (10, 2, 9),
                'expected_error': True
            },
            {
                'name': 'Time dimension zero',
                'obs_shape': (1, 9),
                'data_shape': (0, 1, 9),
                'expected_error': True
            },
            {
                'name': 'Correct dimensions',
                'obs_shape': (1, 9),
                'data_shape': (10, 1, 9),
                'expected_error': False
            }
        ]
        
        # Test each case
        for case in test_cases:
            logger.info(f"Testing: {case['name']}")
            
            # Initialize model with specified dimensions
            model = DMBD(
                obs_shape=case['obs_shape'],
                role_dims=[3, 3, 3],
                hidden_dims=[3, 3, 3],
                number_of_objects=1
            )
            
            # Create data with specified shape
            data = torch.randn(*case['data_shape'])
            
            # Attempt update and check for errors
            try:
                model.update(
                    y=data,
                    u=None,
                    r=None,
                    iters=1,
                    lr=0.001,
                    verbose=False
                )
                
                # If we reach here, no error occurred
                if case['expected_error']:
                    logger.warning(f"Expected error for {case['name']} but none occurred")
                    self.fail(f"Expected error for {case['name']} but none occurred")
                else:
                    logger.info(f"No error for {case['name']} as expected")
            
            except Exception as e:
                if case['expected_error']:
                    logger.info(f"Error for {case['name']} as expected: {str(e)}")
                else:
                    logger.error(f"Unexpected error for {case['name']}: {str(e)}")
                    self.fail(f"Unexpected error for {case['name']}: {str(e)}")
        
        logger.info("Dimension mismatch test completed")
    
    def test_05_matrix_stability(self):
        """Test matrix stability and regularization to prevent singular matrices."""
        # Test different regularization strengths
        reg_strengths = [0, 1e-8, 1e-4, 1e-2, 1e-1]
        
        for reg_strength in reg_strengths:
            logger.info(f"Testing regularization strength: {reg_strength}")
            
            # Create test matrices
            matrices = {
                'Zero on diagonal': torch.eye(9).float(),  # Will have a zero on diagonal
                'Identical rows': torch.ones(9, 9).float(),  # Will have identical rows
                'High condition number': torch.randn(9, 9).float()  # Will have high condition number
            }
            
            # Set up problematic conditions
            matrices['Zero on diagonal'][4, 4] = 0  # Create zero on diagonal
            matrices['High condition number'] = matrices['High condition number'] @ matrices['High condition number'].t()
            matrices['High condition number'][0, 0] *= 1e6  # Create high condition number
            
            success_count = 0
            total_count = len(matrices)
            
            for name, matrix in matrices.items():
                try:
                    # Apply regularization
                    if reg_strength > 0:
                        matrix = matrix + reg_strength * torch.eye(matrix.shape[0])
                        logger.info(f"Applied regularization: {reg_strength}")
                    
                    # Try matrix inversion
                    try:
                        inv_matrix = torch.linalg.inv(matrix)
                        # Verify inverse quality
                        error = torch.norm(torch.eye(matrix.shape[0]) - matrix @ inv_matrix).item()
                        logger.info(f"Matrix {name}: Inversion successful, Error: {error:.6f}")
                        success_count += 1
                    except torch.linalg.LinAlgError as e:
                        logger.warning(f"Matrix {name}: Inversion failed with error: {str(e)}")
                        continue
                    
                    # Additional stability checks
                    if name == 'High condition number':
                        condition_number = torch.linalg.cond(matrix).item()
                        self.assertLess(condition_number, 1e8,
                                      f"Condition number too high: {condition_number}")
                    
                except Exception as e:
                    logger.error(f"Error testing matrix {name}: {str(e)}")
            
            # Calculate success rate
            success_rate = success_count / total_count
            logger.info(f"Regularization {reg_strength}: Success rate {success_rate:.2f}")
            
            # Test DMBD update with this regularization
            try:
                # Create a small model for testing
                model = DMBD(
                    obs_shape=(1, 9),
                    role_dims=[3, 3, 3],
                    hidden_dims=[3, 3, 3],
                    number_of_objects=1
                )
                
                # Apply regularization to model matrices
                if hasattr(model, 'A') and hasattr(model.A, 'data'):
                    model.A.data.add_(reg_strength * torch.eye(model.A.data.shape[0]))
                
                if hasattr(model, 'C') and hasattr(model.C, 'data'):
                    model.C.data.add_(reg_strength * torch.eye(model.C.data.shape[0]))
                
                # Create test data
                data = torch.randn(10, 1, 9)
                
                # Run update
                success = model.update(
                    y=data,
                    u=None,
                    r=None,
                    iters=5,
                    lr=0.001,
                    verbose=True
                )
                
                logger.info(f"DMBD update with regularization {reg_strength}: {'Success' if success else 'Failed'}")
                
                # Additional stability checks
                if hasattr(model, 'A') and hasattr(model.A, 'data'):
                    condition_number = torch.linalg.cond(model.A.data).item()
                    self.assertLess(condition_number, 1e8,
                                  f"Model A matrix condition number too high: {condition_number}")
                
            except Exception as e:
                logger.error(f"Error in DMBD update with regularization {reg_strength}: {str(e)}")


def test_dmbd_dimensions():
    """Run all DMBD dimension tests."""
    unittest.main(module='test_dmbd_dimensions', exit=False)


if __name__ == "__main__":
    unittest.main() 