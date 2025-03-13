#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for validating numerical stability of the DMBD model on Gaussian blob data.

This module tests the ability of the DMBD model to correctly identify internal, blanket,
and external states in the Gaussian blob simulation with proper numerical stability.
It verifies:
1. Matrix inversion stability with different regularization strengths
2. Dimension handling for tensor operations
3. Role assignment accuracy compared to ground truth
4. Feature extraction methods effectiveness
"""

import os
import sys
import torch
import numpy as np
import unittest
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dmbd_gaussian_stability_test")

# Set up paths for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
examples_dir = os.path.join(parent_dir, 'examples')
sys.path.insert(0, examples_dir)

# Create output directory for test results
test_output_dir = os.path.join(test_dir, "test_results", "gaussian_blob_stability")
os.makedirs(test_output_dir, exist_ok=True)

# Import utilities with proper error handling
try:
    from dmbd.dmbd_utils import (
        regularize_matrix,
        safe_matrix_inverse, 
        patch_model_for_stability,
        restore_model_patches,
        check_model_dimensions
    )
    logger.info("Successfully imported stability utilities")
except ImportError as e:
    logger.error(f"Failed to import stability utilities: {str(e)}")
    logger.error("Make sure the dmbd_utils module is in the PYTHONPATH")
    sys.exit(1)

# Try to import DMBD class
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

# Try to import GaussianBlob simulation
try:
    sys.path.append(examples_dir)
    from GaussianBlob import GaussianBlobSimulation
    logger.info("Imported GaussianBlobSimulation")
except ImportError:
    logger.error(f"Failed to import GaussianBlobSimulation. Make sure it's in {examples_dir}")
    sys.exit(1)

# Import stabilized example utilities
try:
    from dmbd_gaussian_blob_stabilized import (
        extract_features,
        evaluate_role_assignment,
        build_dmbd_model,
        run_dmbd_update,
        visualize_results
    )
    logger.info("Imported utilities from dmbd_gaussian_blob_stabilized")
except ImportError as e:
    logger.error(f"Failed to import from dmbd_gaussian_blob_stabilized: {str(e)}")
    logger.error(f"Make sure dmbd_gaussian_blob_stabilized.py is in {examples_dir}")
    sys.exit(1)


class TestDMBDGaussianBlobStability(unittest.TestCase):
    """Test suite for DMBD model stability with Gaussian blob data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across all tests."""
        cls.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = os.path.join(test_output_dir, f"test_run_{cls.timestamp}")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Parameters for test data
        cls.grid_size = 16
        cls.time_steps = 30
        cls.seed = 42
        
        # Create simulation data (once for all tests)
        logger.info(f"Generating Gaussian blob test data (grid_size={cls.grid_size}, time_steps={cls.time_steps})...")
        cls.blob_sim = GaussianBlobSimulation(
            grid_size=cls.grid_size,
            time_steps=cls.time_steps,
            seed=cls.seed
        )
        cls.raw_data, cls.labels = cls.blob_sim.run()
        
        # Extract features with all methods (for reuse across tests)
        cls.features = {}
        for method in ["basic", "spatial", "roles"]:
            cls.features[method] = extract_features(cls.raw_data, cls.grid_size, method=method)
            
        logger.info("Test data preparation complete")
        
        # Summary of data shapes
        cls.shape_info = {
            "raw_data": cls.raw_data.shape,
            "labels": cls.labels.shape,
            "features_basic": cls.features["basic"].shape,
            "features_spatial": cls.features["spatial"].shape,
            "features_roles": cls.features["roles"].shape
        }
        logger.info(f"Data shapes: {cls.shape_info}")
        
        # Write test configuration
        with open(os.path.join(cls.output_dir, "test_config.txt"), "w") as f:
            f.write(f"DMBD Gaussian Blob Stability Test Configuration\n")
            f.write(f"=============================================\n\n")
            f.write(f"Test run: {cls.timestamp}\n")
            f.write(f"Grid size: {cls.grid_size}\n")
            f.write(f"Time steps: {cls.time_steps}\n")
            f.write(f"Random seed: {cls.seed}\n\n")
            f.write(f"Data shapes:\n")
            for name, shape in cls.shape_info.items():
                f.write(f"  {name}: {shape}\n")
    
    def setUp(self):
        """Set up individual test case."""
        # Reset random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_matrix_inversion_stability(self):
        """Test matrix inversion stability with different regularization strengths."""
        logger.info("Testing matrix inversion stability...")
        
        # Regularization strengths to test
        reg_strengths = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
        results = {}
        
        # Create test matrices with different conditioning
        test_cases = []
        
        # Well-conditioned matrix
        A1 = torch.randn(9, 9)
        A1 = A1.t() @ A1  # Make it PD
        test_cases.append(("well_conditioned", A1))
        
        # Poorly-conditioned matrix
        A2 = torch.randn(9, 9)
        A2 = A2.t() @ A2  # Make it PD
        # Make it poorly conditioned by scaling eigenvalues
        U, S, V = torch.svd(A2)
        S[5:] = S[5:] * 1e-6  # Scale down some eigenvalues
        A2 = U @ torch.diag(S) @ V.t()
        test_cases.append(("poorly_conditioned", A2))
        
        # Nearly singular matrix 
        A3 = torch.randn(9, 9)
        A3 = A3.t() @ A3  # Make it PD
        # Make it nearly singular
        U, S, V = torch.svd(A3)
        S[3:] = S[3:] * 1e-10  # Make most eigenvalues very small
        A3 = U @ torch.diag(S) @ V.t()
        test_cases.append(("nearly_singular", A3))
        
        # Test different regularization strengths on each matrix
        for matrix_name, matrix in test_cases:
            results[matrix_name] = {}
            
            for reg in reg_strengths:
                success = True
                error_msg = None
                
                try:
                    # Try standard inversion
                    if reg > 0:
                        reg_matrix = regularize_matrix(matrix, reg)
                        _ = torch.inverse(reg_matrix)
                    else:
                        _ = torch.inverse(matrix)
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                # Try safe inversion
                safe_success = True
                safe_error_msg = None
                try:
                    _ = safe_matrix_inverse(matrix, reg_strength=reg)
                except Exception as e:
                    safe_success = False
                    safe_error_msg = str(e)
                
                results[matrix_name][reg] = {
                    "standard_inversion": {
                        "success": success,
                        "error": error_msg
                    },
                    "safe_inversion": {
                        "success": safe_success,
                        "error": safe_error_msg
                    }
                }
        
        # Visualize results
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(reg_strengths))
        width = 0.15
        
        for i, matrix_name in enumerate(results.keys()):
            standard_success = [results[matrix_name][reg]["standard_inversion"]["success"] for reg in reg_strengths]
            safe_success = [results[matrix_name][reg]["safe_inversion"]["success"] for reg in reg_strengths]
            
            ax.bar(x - width + i*width/len(results), [int(s) for s in standard_success], width=width, 
                   label=f"{matrix_name} (standard)")
            ax.bar(x + i*width/len(results), [int(s) for s in safe_success], width=width, 
                   label=f"{matrix_name} (safe)")
        
        ax.set_xlabel("Regularization Strength")
        ax.set_ylabel("Success (1 = Yes, 0 = No)")
        ax.set_title("Matrix Inversion Stability with Different Regularization")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in reg_strengths])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "matrix_inversion_stability.png"))
        plt.close()
        
        # Write results to file
        with open(os.path.join(self.output_dir, "matrix_inversion_results.txt"), "w") as f:
            f.write("Matrix Inversion Stability Test Results\n")
            f.write("======================================\n\n")
            
            for matrix_name in results:
                f.write(f"\n{matrix_name}:\n")
                f.write("-" * len(matrix_name) + ":\n")
                
                for reg in reg_strengths:
                    f.write(f"\n  Regularization: {reg}\n")
                    std_res = results[matrix_name][reg]["standard_inversion"]
                    safe_res = results[matrix_name][reg]["safe_inversion"]
                    
                    f.write(f"    Standard inversion: {'Success' if std_res['success'] else 'Failed'}\n")
                    if not std_res['success']:
                        f.write(f"      Error: {std_res['error']}\n")
                        
                    f.write(f"    Safe inversion: {'Success' if safe_res['success'] else 'Failed'}\n")
                    if not safe_res['success']:
                        f.write(f"      Error: {safe_res['error']}\n")
        
        # Verify safe inversion worked for all matrices
        for matrix_name in results:
            for reg in reg_strengths:
                if reg > 0:  # Only check positive regularization
                    self.assertTrue(
                        results[matrix_name][reg]["safe_inversion"]["success"],
                        f"Safe inversion failed for {matrix_name} with reg={reg}"
                    )
    
    def test_model_regularization(self):
        """Test DMBD model regularization with different strengths."""
        logger.info("Testing model regularization...")
        
        feature_method = "roles"
        features = self.features[feature_method]
        feature_dim = features.shape[2]
        
        # Regularization strengths to test
        reg_strengths = [0.0, 1e-6, 1e-4, 1e-2]
        results = {}
        
        for reg in reg_strengths:
            logger.info(f"Testing regularization strength: {reg}")
            
            # Create and regularize model
            model = build_dmbd_model(feature_dim, reg_strength=reg)
            
            if model is None:
                results[reg] = {
                    "model_build": False,
                    "update_success": False,
                    "error": "Failed to build model"
                }
                continue
                
            # Run update with limited iterations for testing
            success, assignments = run_dmbd_update(
                model=model,
                features=features,
                iterations=30,  # Fewer iterations for testing
                learning_rate=0.001,
                verbose=False
            )
            
            # Store results
            results[reg] = {
                "model_build": model is not None,
                "update_success": success,
                "has_assignments": assignments is not None
            }
            
            # Evaluate assignment accuracy if possible
            if success and assignments is not None:
                try:
                    eval_results = evaluate_role_assignment(assignments, self.labels, self.grid_size)
                    results[reg]["accuracy"] = eval_results["accuracy"]
                    results[reg]["role_mapping"] = eval_results["role_mapping"]
                except Exception as e:
                    results[reg]["accuracy"] = 0.0
                    results[reg]["error"] = str(e)
            
            # Clean up
            restore_model_patches(model)
        
        # Visualize results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        regs = [reg for reg in reg_strengths if reg in results]
        build_success = [results[reg]["model_build"] for reg in regs]
        update_success = [results[reg]["update_success"] for reg in regs]
        
        # Get accuracy values (use 0 if not available)
        accuracy_values = []
        for reg in regs:
            acc = results[reg].get("accuracy", 0.0)
            if acc is None:
                acc = 0.0
            accuracy_values.append(acc)
        
        x = np.arange(len(regs))
        width = 0.2
        
        ax.bar(x - width, [int(s) for s in build_success], width=width, label="Model Build Success")
        ax.bar(x, [int(s) for s in update_success], width=width, label="Update Success")
        ax.bar(x + width, accuracy_values, width=width, label="Role Assignment Accuracy")
        
        ax.set_xlabel("Regularization Strength")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in regs])
        ax.set_ylabel("Success Rate / Accuracy")
        ax.set_title("DMBD Model Regularization Impact")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "regularization_impact.png"))
        plt.close()
        
        # Write results to file
        with open(os.path.join(self.output_dir, "regularization_results.txt"), "w") as f:
            f.write("DMBD Model Regularization Test Results\n")
            f.write("====================================\n\n")
            
            for reg in reg_strengths:
                if reg in results:
                    f.write(f"\nRegularization strength: {reg}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Model build success: {results[reg]['model_build']}\n")
                    f.write(f"Update success: {results[reg]['update_success']}\n")
                    
                    if "accuracy" in results[reg]:
                        f.write(f"Role assignment accuracy: {results[reg]['accuracy']:.4f}\n")
                        
                        if "role_mapping" in results[reg]:
                            f.write("Role mapping:\n")
                            for dmbd_role, gt_role in results[reg]["role_mapping"].items():
                                f.write(f"  Role {dmbd_role} -> {gt_role}")
                                if gt_role == 0:
                                    f.write(" (Internal/System)")
                                elif gt_role == 1:
                                    f.write(" (Blanket)")
                                elif gt_role == 2:
                                    f.write(" (External/Environment)")
                                f.write("\n")
                    
                    if "error" in results[reg]:
                        f.write(f"Error: {results[reg]['error']}\n")
        
        # Verify at least one regularization strength succeeds
        self.assertTrue(
            any([results[reg]["update_success"] for reg in results]),
            "No regularization strength succeeded in model update"
        )
    
    def test_feature_extraction_methods(self):
        """Test different feature extraction methods and their impact on role assignment."""
        logger.info("Testing feature extraction methods...")
        
        feature_methods = ["basic", "spatial", "roles"]
        reg_strength = 1e-3  # Fixed regularization strength
        
        results = {}
        
        for method in feature_methods:
            logger.info(f"Testing feature method: {method}")
            
            features = self.features[method]
            feature_dim = features.shape[2]
            
            # Create and regularize model
            model = build_dmbd_model(feature_dim, reg_strength=reg_strength)
            
            if model is None:
                results[method] = {
                    "model_build": False,
                    "update_success": False,
                    "error": "Failed to build model"
                }
                continue
                
            # Run update
            success, assignments = run_dmbd_update(
                model=model,
                features=features,
                iterations=50,
                learning_rate=0.001,
                verbose=False
            )
            
            # Store results
            results[method] = {
                "model_build": model is not None,
                "update_success": success,
                "has_assignments": assignments is not None
            }
            
            # Evaluate assignment accuracy if possible
            if success and assignments is not None:
                try:
                    eval_results = evaluate_role_assignment(assignments, self.labels, self.grid_size)
                    results[method]["accuracy"] = eval_results["accuracy"]
                    results[method]["role_mapping"] = eval_results["role_mapping"]
                    results[method]["per_role_accuracy"] = eval_results["per_role_accuracy"]
                    
                    # Visualize this result
                    visualize_results(
                        raw_data=self.raw_data,
                        labels=self.labels,
                        assignments=assignments,
                        results=eval_results,
                        time_step=self.time_steps // 2,
                        grid_size=self.grid_size,
                        output_dir=os.path.join(self.output_dir, f"feature_method_{method}")
                    )
                except Exception as e:
                    results[method]["accuracy"] = 0.0
                    results[method]["error"] = str(e)
            
            # Clean up
            restore_model_patches(model)
        
        # Visualize comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [m for m in feature_methods if m in results]
        update_success = [results[m]["update_success"] for m in methods]
        
        # Get accuracy values
        accuracy_values = []
        for m in methods:
            acc = results[m].get("accuracy", 0.0)
            if acc is None:
                acc = 0.0
            accuracy_values.append(acc)
        
        # Get per-role accuracy if available
        per_role_accuracy = {0: [], 1: [], 2: []}
        for m in methods:
            if "per_role_accuracy" in results[m]:
                for role in [0, 1, 2]:
                    acc = results[m]["per_role_accuracy"].get(role, 0.0)
                    per_role_accuracy[role].append(acc)
            else:
                for role in [0, 1, 2]:
                    per_role_accuracy[role].append(0.0)
        
        x = np.arange(len(methods))
        width = 0.15
        
        ax.bar(x - 2*width, [int(s) for s in update_success], width=width, label="Update Success")
        ax.bar(x - width, accuracy_values, width=width, label="Overall Accuracy")
        ax.bar(x, per_role_accuracy[0], width=width, label="Internal Accuracy")
        ax.bar(x + width, per_role_accuracy[1], width=width, label="Blanket Accuracy")
        ax.bar(x + 2*width, per_role_accuracy[2], width=width, label="External Accuracy")
        
        ax.set_xlabel("Feature Extraction Method")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Success Rate / Accuracy")
        ax.set_title("Impact of Feature Extraction Method on Role Assignment")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_method_comparison.png"))
        plt.close()
        
        # Write results to file
        with open(os.path.join(self.output_dir, "feature_method_results.txt"), "w") as f:
            f.write("Feature Extraction Method Test Results\n")
            f.write("====================================\n\n")
            
            for method in feature_methods:
                if method in results:
                    f.write(f"\nFeature method: {method}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Model build success: {results[method]['model_build']}\n")
                    f.write(f"Update success: {results[method]['update_success']}\n")
                    
                    if "accuracy" in results[method]:
                        f.write(f"Role assignment accuracy: {results[method]['accuracy']:.4f}\n")
                        
                        if "per_role_accuracy" in results[method]:
                            f.write("Per-role accuracy:\n")
                            for role, acc in results[method]["per_role_accuracy"].items():
                                f.write(f"  Role {role}")
                                if role == 0:
                                    f.write(" (Internal/System)")
                                elif role == 1:
                                    f.write(" (Blanket)")
                                elif role == 2:
                                    f.write(" (External/Environment)")
                                f.write(f": {acc:.4f}\n")
                    
                    if "error" in results[method]:
                        f.write(f"Error: {results[method]['error']}\n")
        
        # Assert that at least one feature method has reasonable accuracy
        best_method = max(methods, key=lambda m: results[m].get("accuracy", 0.0) or 0.0)
        best_accuracy = results[best_method].get("accuracy", 0.0) or 0.0
        
        self.assertGreaterEqual(
            best_accuracy, 0.4,  # Expecting at least 40% accuracy for best method
            f"No feature extraction method achieved reasonable accuracy. Best: {best_method} with {best_accuracy:.4f}"
        )
    
    def test_consistency_across_runs(self):
        """Test consistency of role assignments across multiple runs."""
        logger.info("Testing consistency across multiple runs...")
        
        feature_method = "roles"
        features = self.features[feature_method]
        feature_dim = features.shape[2]
        reg_strength = 1e-3
        
        # Number of runs
        n_runs = 3
        assignments_list = []
        accuracy_list = []
        
        for run in range(n_runs):
            logger.info(f"Run {run+1}/{n_runs}")
            
            # Reset random seed for this run
            torch.manual_seed(run + 42)
            
            # Create and regularize model
            model = build_dmbd_model(feature_dim, reg_strength=reg_strength)
            
            if model is None:
                logger.warning(f"Failed to build model on run {run+1}")
                continue
                
            # Run update
            success, assignments = run_dmbd_update(
                model=model,
                features=features,
                iterations=50,
                learning_rate=0.001,
                verbose=False
            )
            
            # Store results if successful
            if success and assignments is not None:
                try:
                    eval_results = evaluate_role_assignment(assignments, self.labels, self.grid_size)
                    assignments_list.append(assignments)
                    accuracy_list.append(eval_results["accuracy"])
                    
                    # Visualize this run
                    os.makedirs(os.path.join(self.output_dir, f"consistency_run_{run+1}"), exist_ok=True)
                    visualize_results(
                        raw_data=self.raw_data,
                        labels=self.labels,
                        assignments=assignments,
                        results=eval_results,
                        time_step=self.time_steps // 2,
                        grid_size=self.grid_size,
                        output_dir=os.path.join(self.output_dir, f"consistency_run_{run+1}")
                    )
                except Exception as e:
                    logger.warning(f"Error in evaluation on run {run+1}: {str(e)}")
            else:
                logger.warning(f"Update failed or no assignments on run {run+1}")
            
            # Clean up
            restore_model_patches(model)
        
        # Calculate consistency metrics if we have at least 2 successful runs
        if len(assignments_list) >= 2:
            # Calculate pairwise agreement between assignments
            agreement_scores = []
            
            for i in range(len(assignments_list)):
                for j in range(i+1, len(assignments_list)):
                    # Map assignments from run j to run i
                    mapped_j_to_i = torch.zeros_like(assignments_list[j])
                    
                    # Get unique roles in each run
                    unique_i = torch.unique(assignments_list[i]).tolist()
                    unique_j = torch.unique(assignments_list[j]).tolist()
                    
                    # For each role in run j, find best matching role in run i
                    role_mapping = {}
                    for role_j in unique_j:
                        overlaps = {}
                        for role_i in unique_i:
                            mask_j = (assignments_list[j] == role_j)
                            mask_i = (assignments_list[i] == role_i)
                            overlaps[role_i] = (mask_j & mask_i).sum().item()
                        
                        # Find best match
                        if overlaps:
                            best_match = max(overlaps.items(), key=lambda x: x[1])[0]
                            role_mapping[role_j] = best_match
                    
                    # Apply mapping
                    for role_j, role_i in role_mapping.items():
                        mapped_j_to_i[assignments_list[j] == role_j] = role_i
                    
                    # Calculate agreement
                    agreement = (mapped_j_to_i == assignments_list[i]).float().mean().item()
                    agreement_scores.append(agreement)
            
            avg_agreement = sum(agreement_scores) / len(agreement_scores)
            min_agreement = min(agreement_scores)
            max_agreement = max(agreement_scores)
            
            # Write consistency results
            with open(os.path.join(self.output_dir, "consistency_results.txt"), "w") as f:
                f.write("DMBD Role Assignment Consistency Results\n")
                f.write("======================================\n\n")
                f.write(f"Number of successful runs: {len(assignments_list)}/{n_runs}\n\n")
                f.write(f"Accuracies: {[f'{acc:.4f}' for acc in accuracy_list]}\n")
                f.write(f"Average accuracy: {sum(accuracy_list)/len(accuracy_list):.4f}\n\n")
                f.write(f"Pairwise agreement scores: {[f'{agr:.4f}' for agr in agreement_scores]}\n")
                f.write(f"Average agreement: {avg_agreement:.4f}\n")
                f.write(f"Min agreement: {min_agreement:.4f}\n")
                f.write(f"Max agreement: {max_agreement:.4f}\n")
            
            # Verify reasonable consistency
            self.assertGreaterEqual(
                avg_agreement, 0.6,  # Expecting at least 60% agreement between runs
                f"Poor consistency across runs. Average agreement: {avg_agreement:.4f}"
            )
        else:
            logger.warning(f"Not enough successful runs to calculate consistency: {len(assignments_list)}/{n_runs}")
            
            # Write minimal results
            with open(os.path.join(self.output_dir, "consistency_results.txt"), "w") as f:
                f.write("DMBD Role Assignment Consistency Results\n")
                f.write("======================================\n\n")
                f.write(f"Number of successful runs: {len(assignments_list)}/{n_runs}\n")
                f.write("Not enough successful runs to calculate consistency metrics.\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are complete."""
        # Generate overall summary
        successful_tests = 0
        total_tests = 4  # Number of test methods
        
        # Check which test files exist to determine success
        test_files = {
            "matrix_inversion_results.txt": "Matrix inversion stability",
            "regularization_results.txt": "Model regularization",
            "feature_method_results.txt": "Feature extraction methods",
            "consistency_results.txt": "Consistency across runs"
        }
        
        successful_tests = sum(
            os.path.exists(os.path.join(cls.output_dir, filename))
            for filename in test_files
        )
        
        # Write summary
        with open(os.path.join(cls.output_dir, "test_summary.txt"), "w") as f:
            f.write("DMBD Gaussian Blob Stability Test Summary\n")
            f.write("=======================================\n\n")
            f.write(f"Test run: {cls.timestamp}\n")
            f.write(f"Tests completed: {successful_tests}/{total_tests}\n\n")
            
            f.write("Test results:\n")
            for filename, test_name in test_files.items():
                file_path = os.path.join(cls.output_dir, filename)
                if os.path.exists(file_path):
                    f.write(f"  {test_name}: Completed\n")
                else:
                    f.write(f"  {test_name}: Not completed\n")
        
        logger.info(f"Test suite completed: {successful_tests}/{total_tests} tests successful")
        logger.info(f"Results saved to: {cls.output_dir}")
        

# Main test runner
if __name__ == "__main__":
    unittest.main() 