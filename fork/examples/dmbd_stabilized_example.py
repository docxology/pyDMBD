#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating a stabilized DMBD model with improved matrix handling.

This script shows how to use the dmbd_utils module to create a more robust DMBD model
that can handle dimension mismatches and singular matrices during inversion.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dmbd_stabilized")

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))  # Add the root directory

# Import DMBD modules
from dmbd.dmbd_utils import (
    regularize_matrix, 
    safe_matrix_inverse, 
    apply_model_regularization,
    patch_model_for_stability,
    restore_model_patches,
    check_model_dimensions,
    debug_tensor
)

# Try to import DMBD class from both possible locations
try:
    from dmbd.dmbd import DMBD
    logger.info("Imported DMBD from dmbd.dmbd")
except ImportError:
    try:
        from DynamicMarkovBlanketDiscovery import DMBD
        logger.info("Imported DMBD from DynamicMarkovBlanketDiscovery")
    except ImportError:
        logger.error("Failed to import DMBD module. Make sure it's in the PYTHONPATH.")
        sys.exit(1)

# Create output directory
output_dir = os.path.join(script_dir, "stabilized_outputs")
os.makedirs(output_dir, exist_ok=True)


def create_test_data(time_steps=20, feature_dim=12, batch_size=1):
    """Create test data tensor with the given dimensions."""
    data = torch.randn(time_steps, batch_size, feature_dim)
    return data


def create_singular_matrices():
    """Create examples of singular or nearly singular matrices."""
    matrices = []
    
    # Case 1: Matrix with zeros on diagonal
    m1 = torch.eye(5)
    m1[2, 2] = 0  # Make one diagonal element zero
    matrices.append(("Zero diagonal", m1))
    
    # Case 2: Matrix with linearly dependent rows
    m2 = torch.randn(5, 5)
    m2[1, :] = m2[0, :]  # Make row 1 identical to row 0
    matrices.append(("Linearly dependent", m2))
    
    # Case 3: Near-singular matrix (high condition number)
    u, s, v = torch.svd(torch.randn(5, 5))
    s[-1] = s[0] * 1e-10  # Make smallest singular value very small
    m3 = u @ torch.diag(s) @ v.t()
    matrices.append(("High condition number", m3))
    
    return matrices


def test_matrix_regularization():
    """Test regularization and safe inversion with singular matrices."""
    logger.info("Testing matrix regularization techniques...")
    
    # Get sample singular matrices
    matrices = create_singular_matrices()
    
    # Test different regularization strengths
    reg_strengths = [0, 1e-8, 1e-4, 1e-2]
    
    # Results table
    results = []
    
    # Test each matrix with each regularization strength
    for name, matrix in matrices:
        logger.info(f"Testing matrix: {name}")
        
        for strength in reg_strengths:
            try:
                if strength == 0:
                    # Try standard inversion
                    inverted = torch.linalg.inv(matrix)
                else:
                    # Try safe inversion with regularization
                    inverted = safe_matrix_inverse(matrix, reg_strength=strength)
                
                # Check if the inversion was successful
                has_nan = torch.isnan(inverted).any().item()
                has_inf = torch.isinf(inverted).any().item()
                
                # Check inversion accuracy
                identity = torch.eye(matrix.shape[0])
                error = torch.norm(inverted @ matrix - identity).item()
                
                results.append({
                    "matrix": name,
                    "reg_strength": strength,
                    "success": not (has_nan or has_inf),
                    "error": error,
                    "has_nan": has_nan,
                    "has_inf": has_inf
                })
                
                logger.info(f"  Reg={strength}: Success={not (has_nan or has_inf)}, Error={error:.6f}")
                
            except Exception as e:
                logger.warning(f"  Reg={strength}: Failed with error: {str(e)}")
                results.append({
                    "matrix": name,
                    "reg_strength": strength,
                    "success": False,
                    "error": float('inf'),
                    "exception": str(e)
                })
    
    # Create visualization of results
    plt.figure(figsize=(12, 8))
    
    # Group results by matrix type
    matrix_types = list(set(r["matrix"] for r in results))
    
    for i, matrix_type in enumerate(matrix_types):
        # Get results for this matrix type
        matrix_results = [r for r in results if r["matrix"] == matrix_type]
        
        # Plot bars for each regularization strength
        x_pos = np.array([j + i*0.3 for j in range(len(reg_strengths))])
        heights = [1 if r.get("success", False) else 0 for r in matrix_results]
        
        # Add error values as text
        for j, r in enumerate(matrix_results):
            if r.get("success", False):
                plt.text(x_pos[j], heights[j] + 0.05, f"{r.get('error', 0):.2e}", 
                        ha='center', va='bottom', rotation=45)
        
        plt.bar(x_pos, heights, width=0.2, label=matrix_type)
    
    plt.xlabel('Regularization Strength')
    plt.ylabel('Success (1=Yes, 0=No)')
    plt.title('Matrix Inversion Success with Regularization')
    plt.xticks(range(len(reg_strengths)), [str(s) for s in reg_strengths])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matrix_regularization_results.png"))
    plt.close()
    
    return results


def test_stabilized_dmbd():
    """Test DMBD model with stabilization techniques."""
    logger.info("Testing stabilized DMBD model...")
    
    # Create test data with various dimensions
    feature_dims = [6, 9, 12]
    time_steps = 20
    
    # Record results
    results = {}
    
    for feature_dim in feature_dims:
        logger.info(f"Testing with feature dimension {feature_dim}")
        data = create_test_data(time_steps=time_steps, feature_dim=feature_dim)
        
        # Create models with and without stabilization
        model_regular = DMBD(
            obs_shape=(1, feature_dim),
            role_dims=[3, 3, 3],  # Three roles with dimension 3 each
            hidden_dims=[3, 3, 3],
            number_of_objects=1
        )
        
        model_stabilized = DMBD(
            obs_shape=(1, feature_dim),
            role_dims=[3, 3, 3],
            hidden_dims=[3, 3, 3],
            number_of_objects=1
        )
        
        # Apply stabilization to the second model
        success = patch_model_for_stability(model_stabilized, reg_strength=1e-3)
        logger.info(f"Applied stabilization: {success}")
        
        # Check dimensions
        dimension_info = check_model_dimensions(model_stabilized)
        if dimension_info["issues"]:
            logger.warning("Dimension issues found:")
            for issue in dimension_info["issues"]:
                logger.warning(f"  - {issue}")
        
        # Try running both models
        try:
            # Regular model
            logger.info("Running regular DMBD model...")
            regular_success = model_regular.update(
                y=data,
                u=None,
                r=None,
                iters=10,
                lr=0.001,
                verbose=True
            )
            
            logger.info(f"Regular model update: {'Success' if regular_success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Regular model failed: {str(e)}")
            regular_success = False
        
        try:
            # Stabilized model
            logger.info("Running stabilized DMBD model...")
            stabilized_success = model_stabilized.update(
                y=data,
                u=None,
                r=None,
                iters=10,
                lr=0.001,
                verbose=True
            )
            
            logger.info(f"Stabilized model update: {'Success' if stabilized_success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Stabilized model failed: {str(e)}")
            stabilized_success = False
        
        # Restore patches
        restore_model_patches(model_stabilized)
        
        # Record results
        results[feature_dim] = {
            "regular_success": regular_success,
            "stabilized_success": stabilized_success
        }
    
    # Create visualization of results
    plt.figure(figsize=(10, 6))
    
    # Plot success rates
    x_pos = np.arange(len(feature_dims))
    regular_heights = [1 if results[dim]["regular_success"] else 0 for dim in feature_dims]
    stabilized_heights = [1 if results[dim]["stabilized_success"] else 0 for dim in feature_dims]
    
    width = 0.35
    plt.bar(x_pos - width/2, regular_heights, width, label='Regular DMBD')
    plt.bar(x_pos + width/2, stabilized_heights, width, label='Stabilized DMBD')
    
    plt.xlabel('Feature Dimension')
    plt.ylabel('Success (1=Yes, 0=No)')
    plt.title('DMBD Model Success with and without Stabilization')
    plt.xticks(x_pos, feature_dims)
    plt.ylim(0, 1.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dmbd_stabilization_results.png"))
    plt.close()
    
    return results


def test_dimension_handling():
    """Test dimension handling capabilities."""
    logger.info("Testing dimension handling...")
    
    # Create test cases with dimension mismatches
    test_cases = [
        {
            "name": "Matching dimensions",
            "obs_shape": (1, 9),
            "data_shape": (20, 1, 9),
            "expected_success": True
        },
        {
            "name": "Feature dimension mismatch",
            "obs_shape": (1, 9),
            "data_shape": (20, 1, 12),
            "expected_success": False
        },
        {
            "name": "Batch dimension mismatch",
            "obs_shape": (1, 9),
            "data_shape": (20, 2, 9),
            "expected_success": False
        }
    ]
    
    # Results storage
    results = []
    
    for case in test_cases:
        logger.info(f"Testing case: {case['name']}")
        
        # Create model
        model = DMBD(
            obs_shape=case["obs_shape"],
            role_dims=[3, 3, 3],
            hidden_dims=[3, 3, 3],
            number_of_objects=1
        )
        
        # Apply stabilization
        patch_model_for_stability(model)
        
        # Create data
        data = torch.randn(*case["data_shape"])
        
        # Try running the model
        try:
            success = model.update(
                y=data,
                u=None,
                r=None,
                iters=5,
                lr=0.001,
                verbose=False
            )
            
            logger.info(f"  Update success: {success}")
            results.append({
                "name": case["name"],
                "success": success,
                "expected": case["expected_success"],
                "error": None
            })
            
        except Exception as e:
            logger.warning(f"  Update failed: {str(e)}")
            results.append({
                "name": case["name"],
                "success": False,
                "expected": case["expected_success"],
                "error": str(e)
            })
        
        # Restore patches
        restore_model_patches(model)
    
    # Create visualization of results
    plt.figure(figsize=(10, 6))
    
    # Plot success vs. expected success
    x_pos = np.arange(len(test_cases))
    actual_heights = [1 if r["success"] else 0 for r in results]
    expected_heights = [1 if r["expected"] else 0 for r in results]
    
    width = 0.35
    plt.bar(x_pos - width/2, actual_heights, width, label='Actual Success')
    plt.bar(x_pos + width/2, expected_heights, width, label='Expected Success')
    
    plt.xlabel('Test Case')
    plt.ylabel('Success (1=Yes, 0=No)')
    plt.title('Dimension Handling Test Results')
    plt.xticks(x_pos, [r["name"] for r in results], rotation=30, ha='right')
    plt.ylim(0, 1.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dimension_handling_results.png"))
    plt.close()
    
    return results


def main():
    """Run all tests and generate a comprehensive report."""
    logger.info("Starting DMBD stabilization tests...")
    
    # Record start time
    start_time = datetime.now()
    
    # Run all tests
    matrix_results = test_matrix_regularization()
    dmbd_results = test_stabilized_dmbd()
    dimension_results = test_dimension_handling()
    
    # Generate comprehensive report
    report_path = os.path.join(output_dir, "stabilization_report.txt")
    with open(report_path, "w") as f:
        f.write("DMBD Stabilization Test Report\n")
        f.write("============================\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {datetime.now() - start_time}\n\n")
        
        # Matrix regularization results
        f.write("1. Matrix Regularization Results\n")
        f.write("-" * 40 + "\n")
        
        for matrix_type in set(r["matrix"] for r in matrix_results):
            f.write(f"\n{matrix_type} Matrix:\n")
            for r in [r for r in matrix_results if r["matrix"] == matrix_type]:
                reg = r["reg_strength"]
                if r["success"]:
                    f.write(f"  Reg={reg}: Success, Error={r.get('error', 'N/A'):.6e}\n")
                else:
                    f.write(f"  Reg={reg}: Failed")
                    if "exception" in r:
                        f.write(f", Error: {r['exception']}\n")
                    else:
                        f.write("\n")
        
        # DMBD stabilization results
        f.write("\n\n2. DMBD Stabilization Results\n")
        f.write("-" * 40 + "\n")
        
        for dim, result in dmbd_results.items():
            f.write(f"\nFeature Dimension {dim}:\n")
            f.write(f"  Regular DMBD: {'Success' if result['regular_success'] else 'Failed'}\n")
            f.write(f"  Stabilized DMBD: {'Success' if result['stabilized_success'] else 'Failed'}\n")
        
        # Dimension handling results
        f.write("\n\n3. Dimension Handling Results\n")
        f.write("-" * 40 + "\n")
        
        for r in dimension_results:
            f.write(f"\n{r['name']}:\n")
            f.write(f"  Expected: {'Success' if r['expected'] else 'Failure'}\n")
            f.write(f"  Actual: {'Success' if r['success'] else 'Failure'}\n")
            if r["error"]:
                f.write(f"  Error: {r['error']}\n")
        
        # Summary and recommendations
        f.write("\n\nSummary and Recommendations\n")
        f.write("-" * 40 + "\n")
        
        # Calculate success rates
        matrix_success_rate = sum(1 for r in matrix_results if r["success"]) / len(matrix_results) * 100
        dmbd_reg_success_rate = sum(1 for dim, r in dmbd_results.items() if r["regular_success"]) / len(dmbd_results) * 100
        dmbd_stab_success_rate = sum(1 for dim, r in dmbd_results.items() if r["stabilized_success"]) / len(dmbd_results) * 100
        dim_success_rate = sum(1 for r in dimension_results if r["success"] == r["expected"]) / len(dimension_results) * 100
        
        f.write(f"Matrix regularization success rate: {matrix_success_rate:.1f}%\n")
        f.write(f"Regular DMBD success rate: {dmbd_reg_success_rate:.1f}%\n")
        f.write(f"Stabilized DMBD success rate: {dmbd_stab_success_rate:.1f}%\n")
        f.write(f"Dimension handling success rate: {dim_success_rate:.1f}%\n\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("1. Use a regularization strength of at least 1e-4 for matrix inversion\n")
        f.write("2. Apply patch_model_for_stability() before running DMBD update\n")
        f.write("3. Verify tensor dimensions match with check_model_dimensions()\n")
        f.write("4. Use safe_matrix_inverse() for all matrix inversions in custom code\n")
    
    logger.info(f"Report generated at {report_path}")
    logger.info("All tests completed.")


if __name__ == "__main__":
    main() 