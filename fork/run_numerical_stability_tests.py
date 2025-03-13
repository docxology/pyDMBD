#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script for DMBD numerical stability tests.

This script provides a convenient interface to run the numerical stability tests
and apply the fixes to the core DMBD implementation.

Usage:
    python run_numerical_stability_tests.py --test-all
    python run_numerical_stability_tests.py --test-matrix-inversion
    python run_numerical_stability_tests.py --test-dimension --apply-fixes
"""

import os
import sys
import argparse
import unittest
import logging
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("numerical_stability_runner")

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def run_tests(args):
    """Run the numerical stability tests."""
    logger.info("Running DMBD numerical stability tests...")
    
    # Import unittest
    try:
        import unittest
        logger.info("Using unittest for test execution")
    except ImportError:
        logger.error("Failed to import unittest module")
        return False
    
    # Import the test module
    try:
        sys.path.insert(0, os.path.join(script_dir, "tests"))
        from test_dmbd_numerical_stability import TestDMBDNumericalStability
        logger.info("Successfully imported test module")
    except ImportError as e:
        logger.error(f"Failed to import test module: {str(e)}")
        return False
    
    # Create test suite based on requested tests
    suite = unittest.TestSuite()
    
    if args.test_all or args.test_matrix_inversion:
        suite.addTest(TestDMBDNumericalStability('test_matrix_conditioning_issues'))
        
    if args.test_all or args.test_dimension:
        suite.addTest(TestDMBDNumericalStability('test_tensor_dimension_consistency'))
        
    if args.test_all or args.test_comprehensive:
        suite.addTest(TestDMBDNumericalStability('test_comprehensive_stability_solution'))
    
    if not suite.countTestCases():
        logger.warning("No tests selected to run. Use --test-all or specific test flags.")
        return False
    
    # Run the tests
    logger.info(f"Running {suite.countTestCases()} tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
        return True
    else:
        logger.error(f"Test run completed with failures: {len(result.failures)} failures, {len(result.errors)} errors")
        return False

def apply_fixes(args):
    """Apply the numerical stability fixes to the core DMBD implementation."""
    if not args.apply_fixes:
        return
    
    logger.info("Applying numerical stability fixes to DMBD core...")
    
    # Check if dmbd module exists
    dmbd_core_dir = os.path.join(script_dir, "dmbd")
    if not os.path.exists(dmbd_core_dir):
        logger.error(f"DMBD core directory not found at {dmbd_core_dir}")
        return False
    
    # Backup original files
    backup_dir = os.path.join(script_dir, "dmbd_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy dmbd core files to backup
    for file in os.listdir(dmbd_core_dir):
        if file.endswith(".py"):
            src = os.path.join(dmbd_core_dir, file)
            dst = os.path.join(backup_dir, file)
            shutil.copy2(src, dst)
    
    logger.info(f"DMBD core files backed up to {backup_dir}")
    
    # Generate patches
    try:
        sys.path.insert(0, os.path.join(script_dir, "tests"))
        from test_dmbd_numerical_stability import TestDMBDNumericalStability
        
        # Create test instance
        test = TestDMBDNumericalStability()
        test.setUp()
        
        # Apply matrix conditioning fixes
        dmbd_file = os.path.join(dmbd_core_dir, "dmbd.py")
        utils_file = os.path.join(dmbd_core_dir, "dmbd_utils.py")
        
        # Add SVD-based pseudo-inverse to dmbd_utils.py
        with open(utils_file, "a") as f:
            f.write("\n\n# Added by numerical stability tests\n")
            f.write("def stable_inverse(tensor):\n")
            f.write("    \"\"\"SVD-based pseudo-inverse that handles ill-conditioned matrices.\"\"\"\n")
            f.write("    # Check if tensor is a singleton\n")
            f.write("    if tensor.numel() == 1:\n")
            f.write("        if tensor.item() == 0:\n")
            f.write("            return torch.tensor([[0.0]], device=tensor.device)\n")
            f.write("        return torch.tensor([[1.0 / tensor.item()]], device=tensor.device)\n")
            f.write("    \n")
            f.write("    # For 1D tensors, convert to diagonal matrix\n")
            f.write("    if tensor.dim() == 1:\n")
            f.write("        tensor = torch.diag(tensor)\n")
            f.write("    \n")
            f.write("    # Use SVD for numerical stability\n")
            f.write("    U, S, V = torch.svd(tensor)\n")
            f.write("    \n")
            f.write("    # Filter small singular values\n")
            f.write("    eps = 1e-6 * S.max()\n")
            f.write("    S_inv = torch.zeros_like(S)\n")
            f.write("    S_inv[S > eps] = 1.0 / S[S > eps]\n")
            f.write("    \n")
            f.write("    # Compute pseudo-inverse\n")
            f.write("    if tensor.shape[0] == tensor.shape[1]:  # Square matrix\n")
            f.write("        return V @ torch.diag(S_inv) @ U.t()\n")
            f.write("    else:  # Non-square matrix\n")
            f.write("        S_inv_mat = torch.zeros(V.shape[1], U.shape[1], device=tensor.device)\n")
            f.write("        for i in range(min(S.shape[0], V.shape[1], U.shape[1])):\n")
            f.write("            S_inv_mat[i, i] = S_inv[i]\n")
            f.write("        return V @ S_inv_mat @ U.t()\n")
        
        logger.info("Added stable_inverse function to dmbd_utils.py")
        
        # Add dimension consistency utilities to dmbd_utils.py
        with open(utils_file, "a") as f:
            f.write("\n\n# Added by numerical stability tests\n")
            f.write("def ensure_batch_consistency(tensors, batch_dim):\n")
            f.write("    \"\"\"Ensures all tensors have consistent batch dimensions.\"\"\"\n")
            f.write("    result = {}\n")
            f.write("    for name, tensor in tensors.items():\n")
            f.write("        if tensor is None or not isinstance(tensor, torch.Tensor):\n")
            f.write("            result[name] = tensor\n")
            f.write("            continue\n")
            f.write("        \n")
            f.write("        if len(tensor.shape) > 0 and tensor.shape[0] != batch_dim:\n")
            f.write("            if tensor.shape[0] == 1:  # Singleton batch\n")
            f.write("                # Broadcast singleton batch to match\n")
            f.write("                result[name] = tensor.expand(batch_dim, *tensor.shape[1:])\n")
            f.write("            elif tensor.shape[0] > batch_dim:  # Too large\n")
            f.write("                # Slice to match\n")
            f.write("                result[name] = tensor[:batch_dim]\n")
            f.write("            else:  # Too small\n")
            f.write("                # Pad by repeating last element\n")
            f.write("                padding = tensor[-1:].expand(batch_dim - tensor.shape[0], *tensor.shape[1:])\n")
            f.write("                result[name] = torch.cat([tensor, padding], dim=0)\n")
            f.write("        else:\n")
            f.write("            result[name] = tensor\n")
            f.write("    \n")
            f.write("    return result\n")
        
        logger.info("Added dimension consistency utilities to dmbd_utils.py")
        
        # Create patches file that contains integration instructions
        patches_dir = os.path.join(script_dir, "tests", "test_results", "numerical_stability")
        os.makedirs(patches_dir, exist_ok=True)
        
        with open(os.path.join(patches_dir, "integration_guide.md"), "w") as f:
            f.write("# DMBD Numerical Stability Integration Guide\n\n")
            f.write("This guide provides instructions for integrating the numerical stability fixes\n")
            f.write("into the core DMBD implementation. The automatic patching has already added some\n")
            f.write("utility functions, but manual integration may be required for some components.\n\n")
            
            f.write("## Matrix Conditioning Improvements\n\n")
            f.write("1. Replace `torch.linalg.inv` calls with `stable_inverse` for improved stability\n")
            f.write("2. Add regularization to covariance matrices before inversion\n")
            f.write("3. Modify the `update_sigma` method to ensure positive definiteness\n\n")
            
            f.write("## Dimension Consistency Fixes\n\n")
            f.write("1. Use `ensure_batch_consistency` before parameter updates\n")
            f.write("2. Add dimension checks in update methods\n")
            f.write("3. Implement broadcasting for matrix operations\n\n")
            
            f.write("## Specific Method Modifications\n\n")
            f.write("### In `update_obs_parms`:\n")
            f.write("```python\n")
            f.write("def update_obs_parms(self):\n")
            f.write("    # Check if latent parameters are initialized\n")
            f.write("    if not hasattr(self, 'mu_r') or self.mu_r is None:\n")
            f.write("        return False\n")
            f.write("        \n")
            f.write("    # Ensure dimensions are consistent\n")
            f.write("    batch_dim = self.Y.shape[0]\n")
            f.write("    tensors = {'mu_r': self.mu_r}\n")
            f.write("    consistent = ensure_batch_consistency(tensors, batch_dim)\n")
            f.write("    self.mu_r = consistent['mu_r']\n")
            f.write("    \n")
            f.write("    # Original implementation continues...\n")
            f.write("```\n\n")
            
            f.write("### In `update_latent_parms`:\n")
            f.write("```python\n")
            f.write("def update_latent_parms(self):\n")
            f.write("    # Ensure dimensions are consistent\n")
            f.write("    if not hasattr(self, 'Y') or self.Y is None:\n")
            f.write("        return False\n")
            f.write("        \n")
            f.write("    batch_dim = self.Y.shape[0]\n")
            f.write("    \n")
            f.write("    # Check and fix tensor dimensions\n")
            f.write("    tensors_to_check = {}\n")
            f.write("    for attr_name in dir(self):\n")
            f.write("        if attr_name.startswith('invSigma_') and isinstance(getattr(self, attr_name), torch.Tensor):\n")
            f.write("            tensors_to_check[attr_name] = getattr(self, attr_name)\n")
            f.write("    \n")
            f.write("    consistent = ensure_batch_consistency(tensors_to_check, batch_dim)\n")
            f.write("    \n")
            f.write("    for name, tensor in consistent.items():\n")
            f.write("        setattr(self, name, tensor)\n")
            f.write("    \n")
            f.write("    # Original implementation continues...\n")
            f.write("```\n")
        
        logger.info(f"Created integration guide at {os.path.join(patches_dir, 'integration_guide.md')}")
        
        # Signal completion
        logger.info("Numerical stability fixes applied and integration guide created")
        return True
        
    except Exception as e:
        logger.error(f"Error applying fixes: {str(e)}")
        return False

def main():
    """Parse command-line arguments and run the requested operations."""
    parser = argparse.ArgumentParser(description="Run DMBD numerical stability tests and apply fixes")
    
    # Test selection
    parser.add_argument("--test-all", action="store_true", help="Run all numerical stability tests")
    parser.add_argument("--test-matrix-inversion", action="store_true", help="Run matrix inversion tests")
    parser.add_argument("--test-dimension", action="store_true", help="Run dimension consistency tests")
    parser.add_argument("--test-comprehensive", action="store_true", help="Run comprehensive stability test")
    
    # Fix application
    parser.add_argument("--apply-fixes", action="store_true", help="Apply fixes to DMBD core")
    
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    
    # Run tests
    success = run_tests(args)
    
    # Apply fixes if requested and tests passed
    if success and args.apply_fixes:
        apply_fixes(args)
    
    # Record and report elapsed time
    elapsed_time = datetime.now() - start_time
    logger.info(f"Total run time: {elapsed_time}")
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 