#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Functionality Test Suite for DMBD

This test suite verifies PyTorch operations and tensor manipulations
that are critical for DMBD analyses.
"""

import os
import sys
import unittest
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('torch_functionality_tests.log')
    ]
)
logger = logging.getLogger("torch_functionality_tests")

class TestTorchFunctionality(unittest.TestCase):
    """Test PyTorch operations and tensor manipulations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = os.path.join(os.path.dirname(__file__), "test_results", "torch_functionality")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create visualization subdirectories
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = {
            "tensor_operations": {},
            "matrix_operations": {},
            "batch_operations": {},
            "numerical_stability": {},
            "device_operations": {},
            "memory_management": {},
            "inference_processes": {},
            "role_assignment": {},
            "markov_blanket_dynamics": {}
        }
        
        # Log setup information
        logger.info(f"Test output directory: {self.output_dir}")
        logger.info(f"Visualization directory: {self.viz_dir}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    def _save_visualization(self, fig, name):
        """Save a matplotlib figure with timestamp and log the location."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(self.viz_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved visualization: {filepath}")
        return filepath
    
    def _plot_tensor_metrics(self, metrics, title):
        """Plot tensor operation metrics with enhanced visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Collect all timing, memory, and stability metrics from test categories
        all_timing = {}
        all_memory = {}
        all_stability = {}
        all_device = {}
        
        # Process each test category and collect metrics
        for test_name, test_metrics in metrics.items():
            if not test_metrics:  # Skip empty metrics
                continue
                
            # Collect timing metrics
            if "timing" in test_metrics:
                for op_name, time_value in test_metrics["timing"].items():
                    all_timing[f"{test_name}_{op_name}"] = time_value
            
            # Collect memory metrics
            if "memory" in test_metrics:
                for mem_name, mem_value in test_metrics["memory"].items():
                    if isinstance(mem_value, (int, float)):  # Only include numeric values
                        all_memory[f"{test_name}_{mem_name}"] = mem_value
            
            # Collect stability metrics
            if "stability" in test_metrics:
                for stab_name, stab_value in test_metrics["stability"].items():
                    all_stability[f"{test_name}_{stab_name}"] = stab_value
                    
            # Collect device metrics
            if "device" in test_metrics:
                for dev_name, dev_value in test_metrics["device"].items():
                    all_device[f"{test_name}_{dev_name}"] = dev_value
        
        # Plot 1: Operation timing
        ax1 = axes[0, 0]
        if all_timing:
            # Sort by value for better visualization
            sorted_timing = dict(sorted(all_timing.items(), key=lambda x: x[1], reverse=True)[:10])
            sns.barplot(x=list(sorted_timing.keys()),
                      y=list(sorted_timing.values()),
                      ax=ax1)
            ax1.set_title("Operation Timing (Top 10)")
            ax1.set_ylabel("Time (ms)")
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, "No timing data available", ha='center', va='center')
            ax1.set_title("Operation Timing")
        
        # Plot 2: Memory usage
        ax2 = axes[0, 1]
        if all_memory:
            # Sort by value for better visualization
            sorted_memory = dict(sorted(all_memory.items(), key=lambda x: x[1], reverse=True)[:10])
            sns.barplot(x=list(sorted_memory.keys()),
                      y=list(sorted_memory.values()),
                      ax=ax2)
            ax2.set_title("Memory Usage (Top 10)")
            ax2.set_ylabel("Memory (MB)")
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, "No memory data available", ha='center', va='center')
            ax2.set_title("Memory Usage")
        
        # Plot 3: Numerical stability
        ax3 = axes[1, 0]
        if all_stability:
            # Sort by value for better visualization
            sorted_stability = dict(sorted(all_stability.items(), key=lambda x: x[1], reverse=True)[:10])
            sns.barplot(x=list(sorted_stability.keys()),
                      y=list(sorted_stability.values()),
                      ax=ax3)
            ax3.set_title("Numerical Stability (Top 10)")
            ax3.set_ylabel("Value")
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, "No stability data available", ha='center', va='center')
            ax3.set_title("Numerical Stability")
        
        # Plot 4: Test metrics summary
        ax4 = axes[1, 1]
        # Count metrics per test category
        metrics_count = {test: len(test_metrics) for test, test_metrics in metrics.items() if test_metrics}
        if metrics_count:
            sns.barplot(x=list(metrics_count.keys()),
                      y=list(metrics_count.values()),
                      ax=ax4)
            ax4.set_title("Metrics Count by Test")
            ax4.set_ylabel("Number of Metrics")
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, "No metrics data available", ha='center', va='center')
            ax4.set_title("Metrics Count by Test")
        
        plt.tight_layout()
        return self._save_visualization(fig, f"tensor_metrics_{title.lower().replace(' ', '_')}")
    
    def _plot_tensor_heatmap(self, tensor, title, ax):
        """Plot tensor heatmap with enhanced visualization."""
        data = tensor.detach().numpy()
        sns.heatmap(data, ax=ax, cmap='viridis', center=0)
        ax.set_title(title)
        return ax
    
    def _plot_tensor_distribution(self, tensor, title, ax):
        """Plot tensor value distribution."""
        data = tensor.detach().numpy().flatten()
        sns.histplot(data, ax=ax, kde=True)
        ax.set_title(title)
        return ax
    
    def _plot_tensor_trajectory(self, tensor, title, ax):
        """Plot tensor trajectory over time."""
        data = tensor.detach().numpy()
        ax.plot(data)
        ax.set_title(title)
        return ax
    
    def test_tensor_operations(self):
        """Test basic tensor operations used in DMBD."""
        logger.info("Testing basic tensor operations...")
        
        # Test tensor creation and reshaping with DMBD-like dimensions
        batch_size = 20
        time_steps = 10
        feature_dim = 9
        role_dim = 3
        
        # Create tensors with DMBD-like shapes
        observations = torch.randn(time_steps, batch_size, feature_dim)
        roles = torch.randn(time_steps, batch_size, role_dim)
        hidden_states = torch.randn(time_steps, batch_size, role_dim * 3)  # 3 for internal, blanket, external
        
        # Test tensor broadcasting with DMBD-like operations
        transition_matrix = torch.randn(role_dim * 3, role_dim * 3)  # State transition matrix
        emission_matrix = torch.randn(feature_dim, role_dim * 3)     # Emission matrix
        
        # Test forward pass operations
        predicted_states = hidden_states @ transition_matrix.T
        predicted_obs = predicted_states @ emission_matrix.T
        
        # Verify shapes
        self.assertEqual(predicted_states.shape, (time_steps, batch_size, role_dim * 3))
        self.assertEqual(predicted_obs.shape, (time_steps, batch_size, feature_dim))
        
        # Test gradient computation with DMBD-like loss
        observations.requires_grad_(True)
        hidden_states.requires_grad_(True)
        
        # Compute MSE loss
        loss = (predicted_obs - observations).pow(2).mean()
        loss.backward()
        
        # Verify gradients
        self.assertIsNotNone(observations.grad)
        self.assertIsNotNone(hidden_states.grad)
        
        # Enhanced visualizations for DMBD operations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("DMBD Tensor Operations Visualization", fontsize=16)
        
        # Plot 1: Observation tensor heatmap
        self._plot_tensor_heatmap(
            observations[0, 0].reshape(3, 3), 
            "Observation Tensor", 
            axes[0, 0]
        )
        
        # Plot 2: Hidden state tensor heatmap
        self._plot_tensor_heatmap(
            hidden_states[0, 0].reshape(-1, 3), 
            "Hidden States", 
            axes[0, 1]
        )
        
        # Plot 3: Transition matrix heatmap
        self._plot_tensor_heatmap(
            transition_matrix,
            "Transition Matrix",
            axes[0, 2]
        )
        
        # Plot 4: Emission matrix heatmap
        self._plot_tensor_heatmap(
            emission_matrix,
            "Emission Matrix",
            axes[1, 0]
        )
        
        # Plot 5: Predicted observations heatmap
        self._plot_tensor_heatmap(
            predicted_obs[0, 0].reshape(3, 3),
            "Predicted Observations",
            axes[1, 1]
        )
        
        # Plot 6: Gradient heatmap
        self._plot_tensor_heatmap(
            observations.grad[0, 0].reshape(3, 3),
            "Observation Gradients",
            axes[1, 2]
        )
        
        plt.tight_layout()
        self._save_visualization(fig, "dmbd_tensor_operations")
        
        # Record detailed metrics
        self.metrics["tensor_operations"] = {
            "shapes": {
                "observations": list(observations.shape),
                "hidden_states": list(hidden_states.shape),
                "transition_matrix": list(transition_matrix.shape),
                "emission_matrix": list(emission_matrix.shape),
                "predicted_obs": list(predicted_obs.shape)
            },
            "memory": {
                "observations": observations.element_size() * observations.nelement() / (1024 * 1024),
                "hidden_states": hidden_states.element_size() * hidden_states.nelement() / (1024 * 1024),
                "total": sum(t.element_size() * t.nelement() for t in [
                    observations, hidden_states, transition_matrix, emission_matrix
                ]) / (1024 * 1024)
            },
            "gradients": {
                "obs_grad_norm": observations.grad.norm().item(),
                "hidden_grad_norm": hidden_states.grad.norm().item(),
                "loss": loss.item()
            },
            "statistics": {
                "obs_mean": observations.mean().item(),
                "obs_std": observations.std().item(),
                "hidden_mean": hidden_states.mean().item(),
                "hidden_std": hidden_states.std().item()
            }
        }
    
    def test_matrix_operations(self):
        """Test matrix operations critical for DMBD."""
        logger.info("Testing matrix operations...")
        
        # Test matrix inversion
        A = torch.randn(9, 9)
        A = A @ A.t() + torch.eye(9)  # Make it positive definite
        try:
            A_inv = torch.linalg.inv(A)
            self.assertEqual(A_inv.shape, (9, 9))
        except Exception as e:
            self.fail(f"Matrix inversion failed: {str(e)}")
        
        # Test SVD decomposition
        U, S, V = torch.svd(A)
        self.assertEqual(U.shape, (9, 9))
        self.assertEqual(S.shape, (9,))
        self.assertEqual(V.shape, (9, 9))
        
        # Test eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        self.assertEqual(eigenvalues.shape, (9,))
        self.assertEqual(eigenvectors.shape, (9, 9))
        
        # Visualize matrix operations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Matrix Operations Visualization", fontsize=16)
        
        # Plot 1: Original matrix
        ax1 = axes[0, 0]
        sns.heatmap(A.detach().numpy(), ax=ax1)
        ax1.set_title("Original Matrix")
        
        # Plot 2: Inverse matrix
        ax2 = axes[0, 1]
        sns.heatmap(A_inv.detach().numpy(), ax=ax2)
        ax2.set_title("Inverse Matrix")
        
        # Plot 3: SVD components
        ax3 = axes[1, 0]
        sns.heatmap(U.detach().numpy(), ax=ax3)
        ax3.set_title("U Matrix (SVD)")
        
        # Plot 4: Eigenvalues
        ax4 = axes[1, 1]
        ax4.plot(eigenvalues.detach().numpy())
        ax4.set_title("Eigenvalues")
        
        plt.tight_layout()
        self._save_visualization(fig, "matrix_operations")
        
        # Record metrics
        self.metrics["matrix_operations"] = {
            "timing": {
                "inversion": 0.1,  # Example timing
                "svd": 0.2,
                "eigenvalue": 0.15
            },
            "stability": {
                "condition_number": torch.linalg.cond(A).item(),
                "svd_condition": torch.linalg.cond(U).item()
            }
        }
    
    def test_batch_operations(self):
        """Test batch operations used in DMBD."""
        logger.info("Testing batch operations...")
        
        # Test batch matrix multiplication
        batch_size = 20
        x = torch.randn(batch_size, 9, 9)  # Batch of matrices
        y = torch.randn(batch_size, 9, 1)  # Batch of vectors
        z = torch.bmm(x, y)  # Batch matrix multiplication
        self.assertEqual(z.shape, (batch_size, 9, 1))
        
        # Test batch-wise operations
        mean = x.mean(dim=0)  # Average across batch
        std = x.std(dim=0)    # Standard deviation across batch
        self.assertEqual(mean.shape, (9, 9))
        self.assertEqual(std.shape, (9, 9))
        
        # Visualize batch operations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Batch Operations Visualization", fontsize=16)
        
        # Plot 1: Batch mean
        ax1 = axes[0, 0]
        sns.heatmap(mean.detach().numpy(), ax=ax1)
        ax1.set_title("Batch Mean")
        
        # Plot 2: Batch std
        ax2 = axes[0, 1]
        sns.heatmap(std.detach().numpy(), ax=ax2)
        ax2.set_title("Batch Standard Deviation")
        
        # Plot 3: Batch multiplication result
        ax3 = axes[1, 0]
        sns.heatmap(z[0].detach().numpy(), ax=ax3)
        ax3.set_title("Batch Multiplication Result")
        
        # Plot 4: Batch statistics
        ax4 = axes[1, 1]
        ax4.boxplot([x[:, i, j].detach().numpy() for i in range(9) for j in range(9)])
        ax4.set_title("Batch Statistics")
        
        plt.tight_layout()
        self._save_visualization(fig, "batch_operations")
        
        # Record metrics
        self.metrics["batch_operations"] = {
            "timing": {
                "bmm": 0.1,  # Example timing
                "mean": 0.05,
                "std": 0.05
            },
            "memory": {
                "batch_size": batch_size,
                "total_elements": x.nelement(),
                "memory_usage": x.element_size() * x.nelement() / (1024 * 1024)
            }
        }
    
    def test_numerical_stability(self):
        """Test numerical stability of operations."""
        logger.info("Testing numerical stability...")
        
        # Test handling of small numbers
        small = torch.tensor(1e-10)
        self.assertFalse(torch.isnan(small))
        
        # Test handling of large numbers
        large = torch.tensor(1e10)
        self.assertFalse(torch.isinf(large))
        
        # Test matrix conditioning
        A = torch.randn(9, 9)
        A = A @ A.t() + torch.eye(9) * 1e-6  # Add small regularization
        try:
            A_inv = torch.linalg.inv(A)
            self.assertFalse(torch.any(torch.isnan(A_inv)))
        except Exception as e:
            self.fail(f"Stable matrix inversion failed: {str(e)}")
        
        # Visualize numerical stability
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Numerical Stability Visualization", fontsize=16)
        
        # Plot 1: Small number handling
        ax1 = axes[0, 0]
        ax1.semilogy([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax1.set_title("Small Number Handling")
        
        # Plot 2: Large number handling
        ax2 = axes[0, 1]
        ax2.semilogy([1, 1e2, 1e4, 1e6, 1e8, 1e10])
        ax2.set_title("Large Number Handling")
        
        # Plot 3: Matrix conditioning
        ax3 = axes[1, 0]
        sns.heatmap(A.detach().numpy(), ax=ax3)
        ax3.set_title("Regularized Matrix")
        
        # Plot 4: Condition number evolution
        ax4 = axes[1, 1]
        condition_numbers = []
        for i in range(10):
            A_test = torch.randn(9, 9)
            A_test = A_test @ A_test.t() + torch.eye(9) * (1e-6 / (i + 1))
            condition_numbers.append(torch.linalg.cond(A_test).item())
        ax4.plot(condition_numbers)
        ax4.set_title("Condition Number Evolution")
        
        plt.tight_layout()
        self._save_visualization(fig, "numerical_stability")
        
        # Record metrics
        self.metrics["numerical_stability"] = {
            "stability": {
                "small_number": small.item(),
                "large_number": large.item(),
                "condition_number": torch.linalg.cond(A).item()
            },
            "timing": {
                "inversion": 0.1,  # Example timing
                "conditioning": 0.05
            }
        }
    
    def test_device_operations(self):
        """Test operations across different devices."""
        logger.info("Testing device operations...")
        
        if torch.cuda.is_available():
            # Test CPU to GPU transfer
            x_cpu = torch.randn(20, 1, 9)
            x_gpu = x_cpu.cuda()
            self.assertEqual(x_gpu.device.type, 'cuda')
            
            # Test GPU operations
            y_gpu = x_gpu @ x_gpu.transpose(-2, -1)
            self.assertEqual(y_gpu.device.type, 'cuda')
            
            # Test GPU to CPU transfer
            y_cpu = y_gpu.cpu()
            self.assertEqual(y_cpu.device.type, 'cpu')
            
            # Visualize device operations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Device Operations Visualization", fontsize=16)
            
            # Plot 1: CPU tensor
            ax1 = axes[0, 0]
            sns.heatmap(x_cpu[0, 0].detach().numpy(), ax=ax1)
            ax1.set_title("CPU Tensor")
            
            # Plot 2: GPU tensor
            ax2 = axes[0, 1]
            sns.heatmap(y_gpu[0, 0].cpu().detach().numpy(), ax=ax2)
            ax2.set_title("GPU Tensor")
            
            # Plot 3: Device transfer timing
            ax3 = axes[1, 0]
            transfer_times = []
            for size in [100, 1000, 10000]:
                x = torch.randn(size, size)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                x_gpu = x.cuda()
                end.record()
                torch.cuda.synchronize()
                transfer_times.append(start.elapsed_time(end))
            ax3.plot([100, 1000, 10000], transfer_times)
            ax3.set_title("Device Transfer Timing")
            
            # Plot 4: Memory usage
            ax4 = axes[1, 1]
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)
            ax4.bar(["GPU Memory"], [memory_usage])
            ax4.set_title("GPU Memory Usage")
            
            plt.tight_layout()
            self._save_visualization(fig, "device_operations")
            
            # Record metrics
            self.metrics["device_operations"] = {
                "timing": {
                    "cpu_to_gpu": transfer_times[0],
                    "gpu_to_cpu": transfer_times[1]
                },
                "memory": {
                    "gpu_allocated": memory_usage,
                    "gpu_cached": torch.cuda.memory_reserved() / (1024 * 1024)
                }
            }
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        logger.info("Testing memory management...")
        
        # Test memory allocation and deallocation
        x = torch.randn(1000, 1000)  # Large tensor
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Perform operations
        y = x @ x.t()
        z = torch.linalg.inv(y)
        
        # Clear memory
        del x, y, z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.assertLessEqual(final_memory, initial_memory)
        
        # Visualize memory management
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Memory Management Visualization", fontsize=16)
        
        # Plot 1: Memory allocation over time
        ax1 = axes[0, 0]
        memory_trace = []
        for i in range(10):
            x = torch.randn(1000, 1000)
            if torch.cuda.is_available():
                memory_trace.append(torch.cuda.memory_allocated() / (1024 * 1024))
            else:
                memory_trace.append(x.element_size() * x.nelement() / (1024 * 1024))
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ax1.plot(memory_trace)
        ax1.set_title("Memory Allocation Over Time")
        
        # Plot 2: Memory cleanup
        ax2 = axes[0, 1]
        cleanup_times = []
        if torch.cuda.is_available():
            for size in [100, 1000, 10000]:
                x = torch.randn(size, size)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                del x
                torch.cuda.empty_cache()
                end.record()
                torch.cuda.synchronize()
                cleanup_times.append(start.elapsed_time(end))
            ax2.plot([100, 1000, 10000], cleanup_times)
        else:
            ax2.text(0.5, 0.5, "CUDA not available", ha='center', va='center')
        ax2.set_title("Memory Cleanup Timing")
        
        # Plot 3: Memory fragmentation
        ax3 = axes[1, 0]
        fragmentation = []
        if torch.cuda.is_available():
            for i in range(5):
                x = torch.randn(1000, 1000)
                y = torch.randn(1000, 1000)
                fragmentation.append(torch.cuda.memory_reserved() / torch.cuda.memory_allocated())
                del x, y
                torch.cuda.empty_cache()
            ax3.plot(fragmentation)
        else:
            ax3.text(0.5, 0.5, "CUDA not available", ha='center', va='center')
        ax3.set_title("Memory Fragmentation")
        
        # Plot 4: Memory usage by operation
        ax4 = axes[1, 1]
        operations = ["Allocation", "Matrix Multiplication", "Inversion", "Cleanup"]
        memory_usage = [
            initial_memory / (1024 * 1024),
            torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            final_memory / (1024 * 1024)
        ]
        ax4.bar(operations, memory_usage)
        ax4.set_title("Memory Usage by Operation")
        
        plt.tight_layout()
        self._save_visualization(fig, "memory_management")
        
        # Record metrics
        self.metrics["memory_management"] = {
            "memory": {
                "initial": initial_memory / (1024 * 1024),
                "final": final_memory / (1024 * 1024),
                "peak": max(memory_trace) if memory_trace else 0
            },
            "timing": {
                "cleanup": cleanup_times[0] if cleanup_times else 0
            }
        }
    
    def test_inference_processes(self):
        """Test PyTorch inference processes relevant to DMBD."""
        logger.info("Testing inference processes...")
        
        # Test forward pass with latent variables
        batch_size = 10
        seq_length = 20
        hidden_dim = 9
        obs_dim = 6
        
        # Create synthetic data
        latents = torch.randn(seq_length, batch_size, hidden_dim)
        observations = torch.randn(seq_length, batch_size, obs_dim)
        
        # Test Kalman filter-style operations
        # Prior
        prior_mean = torch.zeros(hidden_dim)
        prior_cov = torch.eye(hidden_dim)
        
        # Transition matrix (block structure for Markov blanket)
        A = torch.zeros(hidden_dim, hidden_dim)
        s_dim, b_dim, z_dim = 3, 3, 3  # Environment, boundary, object dims
        
        # Set block diagonal and off-diagonal terms
        A[:s_dim, :s_dim] = torch.randn(s_dim, s_dim)  # Environment dynamics
        A[s_dim:s_dim+b_dim, :] = torch.randn(b_dim, hidden_dim)  # Boundary coupling
        A[s_dim+b_dim:, s_dim+b_dim:] = torch.randn(z_dim, z_dim)  # Object dynamics
        
        # Observation matrix
        C = torch.randn(obs_dim, hidden_dim)
        
        # Process and observation noise
        Q = torch.eye(hidden_dim) * 0.1
        R = torch.eye(obs_dim) * 0.1
        
        # Run forward pass
        filtered_means = []
        filtered_covs = []
        
        current_mean = prior_mean
        current_cov = prior_cov
        
        for t in range(seq_length):
            # Predict
            pred_mean = A @ current_mean
            pred_cov = A @ current_cov @ A.t() + Q
            
            # Update
            innovation = observations[t, 0] - C @ pred_mean
            S = C @ pred_cov @ C.t() + R
            K = pred_cov @ C.t() @ torch.linalg.inv(S)
            
            current_mean = pred_mean + K @ innovation
            current_cov = pred_cov - K @ C @ pred_cov
            
            filtered_means.append(current_mean)
            filtered_covs.append(current_cov)
        
        filtered_means = torch.stack(filtered_means)
        filtered_covs = torch.stack(filtered_covs)
        
        # Visualize inference results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Inference Process Visualization", fontsize=16)
        
        # Plot 1: Latent trajectories
        ax1 = axes[0, 0]
        for i in range(min(3, hidden_dim)):
            ax1.plot(filtered_means[:, i].detach().numpy(), label=f'Dim {i}')
        ax1.set_title("Filtered State Trajectories")
        ax1.legend()
        
        # Plot 2: Covariance evolution
        ax2 = axes[0, 1]
        cov_norms = [torch.norm(cov).item() for cov in filtered_covs]
        ax2.plot(cov_norms)
        ax2.set_title("Covariance Matrix Norm Evolution")
        
        # Plot 3: Innovation sequence
        ax3 = axes[1, 0]
        innovations = observations[:, 0] - torch.stack([C @ mean for mean in filtered_means])
        ax3.plot(torch.norm(innovations, dim=1).detach().numpy())
        ax3.set_title("Innovation Magnitude")
        
        # Plot 4: Block structure visualization
        ax4 = axes[1, 1]
        sns.heatmap(A.detach().numpy(), ax=ax4)
        ax4.set_title("Transition Matrix Block Structure")
        
        plt.tight_layout()
        self._save_visualization(fig, "inference_processes")
        
        # Record metrics
        self.metrics["inference_processes"] = {
            "timing": {
                "forward_pass": 0.1,  # Example timing
                "filtering": 0.2
            },
            "stability": {
                "mean_innovation": torch.mean(torch.norm(innovations, dim=1)).item(),
                "final_cov_norm": cov_norms[-1]
            }
        }
        
        # Verify results
        self.assertEqual(filtered_means.shape, (seq_length, hidden_dim))
        self.assertEqual(filtered_covs.shape, (seq_length, hidden_dim, hidden_dim))
        self.assertFalse(torch.any(torch.isnan(filtered_means)))
        self.assertFalse(torch.any(torch.isnan(filtered_covs)))

    def test_role_assignment(self):
        """Test role assignment and Markov blanket structure."""
        logger.info("Testing role assignment...")
        
        # Setup dimensions
        num_objects = 10
        num_timesteps = 30
        role_dims = torch.tensor([2, 2, 2])  # Environment, boundary, object roles
        total_roles = role_dims.sum().item()
        
        # Create synthetic role probabilities
        role_probs = torch.softmax(torch.randn(num_timesteps, num_objects, total_roles), dim=-1)
        
        # Create transition matrix with Markov blanket structure
        trans_matrix = torch.zeros(total_roles, total_roles)
        # Allow transitions within same type
        for i, dim in enumerate(role_dims):
            start_idx = sum(role_dims[:i])
            end_idx = start_idx + dim
            trans_matrix[start_idx:end_idx, start_idx:end_idx] = torch.rand(dim, dim)
        # Allow transitions through boundary
        b_start = role_dims[0]
        b_end = b_start + role_dims[1]
        trans_matrix[b_start:b_end, :] = torch.rand(role_dims[1], total_roles)
        
        # Normalize transition probabilities
        trans_matrix = trans_matrix / trans_matrix.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        # Forward pass through roles
        forward_probs = []
        current_prob = role_probs[0]
        
        for t in range(1, num_timesteps):
            # Propagate probabilities
            pred_prob = current_prob @ trans_matrix
            # Update with observation
            current_prob = pred_prob * role_probs[t]
            current_prob = current_prob / current_prob.sum(dim=1, keepdim=True).clamp(min=1e-10)
            forward_probs.append(current_prob)
        
        forward_probs = torch.stack(forward_probs)
        
        # Visualize role assignments
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Role Assignment Visualization", fontsize=16)
        
        # Plot 1: Role probabilities over time for one object
        ax1 = axes[0, 0]
        obj_idx = 0
        for r in range(total_roles):
            ax1.plot(role_probs[:, obj_idx, r].numpy(), label=f'Role {r}')
        ax1.set_title(f"Role Probabilities (Object {obj_idx})")
        ax1.legend()
        
        # Plot 2: Transition matrix structure
        ax2 = axes[0, 1]
        sns.heatmap(trans_matrix.numpy(), ax=ax2)
        ax2.set_title("Role Transition Matrix")
        
        # Plot 3: Role distribution across objects
        ax3 = axes[1, 0]
        role_dist = role_probs[-1].mean(dim=0)
        ax3.bar(range(total_roles), role_dist.numpy())
        ax3.set_title("Final Role Distribution")
        
        # Plot 4: Role switching frequency
        ax4 = axes[1, 1]
        switches = torch.argmax(role_probs, dim=-1)
        switch_counts = []
        for obj in range(num_objects):
            switches_obj = (switches[1:, obj] != switches[:-1, obj]).sum().item()
            switch_counts.append(switches_obj)
        ax4.bar(range(num_objects), switch_counts)
        ax4.set_title("Role Switching Frequency")
        
        plt.tight_layout()
        self._save_visualization(fig, "role_assignment")
        
        # Record metrics
        self.metrics["role_assignment"] = {
            "timing": {
                "forward_pass": 0.1
            },
            "statistics": {
                "avg_switches": sum(switch_counts) / num_objects,
                "role_entropy": -torch.sum(role_dist * torch.log(role_dist + 1e-10)).item()
            }
        }
        
        # Verify results
        self.assertEqual(forward_probs.shape, (num_timesteps-1, num_objects, total_roles))
        self.assertFalse(torch.any(torch.isnan(forward_probs)))
        self.assertTrue(torch.allclose(forward_probs.sum(dim=-1), torch.ones_like(forward_probs.sum(dim=-1))))

    def test_markov_blanket_dynamics(self):
        """Test Markov blanket dynamics and information flow."""
        logger.info("Testing Markov blanket dynamics...")
        
        # Setup dimensions
        hidden_dim = 9
        s_dim = b_dim = z_dim = 3  # Environment, boundary, object dimensions
        num_timesteps = 50
        
        # Create block matrices for Markov blanket structure
        A_ss = torch.randn(s_dim, s_dim)  # Environment dynamics
        A_sb = torch.randn(s_dim, b_dim)  # Environment-boundary coupling
        A_bs = torch.randn(b_dim, s_dim)  # Boundary-environment coupling
        A_bb = torch.randn(b_dim, b_dim)  # Boundary dynamics
        A_bz = torch.randn(b_dim, z_dim)  # Boundary-object coupling
        A_zb = torch.randn(z_dim, b_dim)  # Object-boundary coupling
        A_zz = torch.randn(z_dim, z_dim)  # Object dynamics
        
        # Construct full transition matrix
        A = torch.zeros(hidden_dim, hidden_dim)
        A[:s_dim, :s_dim] = A_ss
        A[:s_dim, s_dim:s_dim+b_dim] = A_sb
        A[s_dim:s_dim+b_dim, :s_dim] = A_bs
        A[s_dim:s_dim+b_dim, s_dim:s_dim+b_dim] = A_bb
        A[s_dim:s_dim+b_dim, s_dim+b_dim:] = A_bz
        A[s_dim+b_dim:, s_dim:s_dim+b_dim] = A_zb
        A[s_dim+b_dim:, s_dim+b_dim:] = A_zz
        
        # Initialize state
        x0 = torch.randn(hidden_dim)
        states = [x0]
        
        # Simulate dynamics
        for t in range(num_timesteps-1):
            next_state = A @ states[-1] + 0.1 * torch.randn(hidden_dim)
            states.append(next_state)
        
        states = torch.stack(states)
        
        # Calculate information flow metrics
        s_states = states[:, :s_dim]
        b_states = states[:, s_dim:s_dim+b_dim]
        z_states = states[:, s_dim+b_dim:]
        
        # Compute mutual information proxies using correlation
        mi_sb = torch.corrcoef(torch.cat([s_states, b_states], dim=1).t())[:s_dim, s_dim:]
        mi_bz = torch.corrcoef(torch.cat([b_states, z_states], dim=1).t())[:b_dim, b_dim:]
        
        # Visualize dynamics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Markov Blanket Dynamics", fontsize=16)
        
        # Plot 1: State trajectories
        ax1 = axes[0, 0]
        ax1.plot(s_states[:, 0].numpy(), label='Environment')
        ax1.plot(b_states[:, 0].numpy(), label='Boundary')
        ax1.plot(z_states[:, 0].numpy(), label='Object')
        ax1.set_title("State Trajectories")
        ax1.legend()
        
        # Plot 2: Information flow
        ax2 = axes[0, 1]
        sns.heatmap(torch.abs(mi_sb).numpy(), ax=ax2)
        ax2.set_title("Environment-Boundary\nMutual Information")
        
        # Plot 3: Phase space
        ax3 = axes[1, 0]
        ax3.scatter(s_states[:, 0].numpy(), b_states[:, 0].numpy(), alpha=0.5)
        ax3.set_xlabel("Environment")
        ax3.set_ylabel("Boundary")
        ax3.set_title("Phase Space")
        
        # Plot 4: Transition matrix structure
        ax4 = axes[1, 1]
        sns.heatmap(A.numpy(), ax=ax4)
        ax4.set_title("Transition Matrix")
        
        plt.tight_layout()
        self._save_visualization(fig, "markov_blanket_dynamics")
        
        # Record metrics
        self.metrics["markov_blanket_dynamics"] = {
            "timing": {
                "simulation": 0.1
            },
            "information_flow": {
                "avg_mi_sb": torch.abs(mi_sb).mean().item(),
                "avg_mi_bz": torch.abs(mi_bz).mean().item()
            }
        }
        
        # Verify results
        self.assertEqual(states.shape, (num_timesteps, hidden_dim))
        self.assertFalse(torch.any(torch.isnan(states)))
        self.assertTrue(torch.all(torch.isfinite(states)))

    def tearDown(self):
        """Clean up after tests and generate final report."""
        # Generate final visualization of all metrics
        metrics_viz_path = self._plot_tensor_metrics(self.metrics, "PyTorch Functionality Test Results")
        logger.info(f"Generated final metrics visualization: {metrics_viz_path}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.output_dir, "torch_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")
        
        # Log summary of all visualizations
        logger.info("Test Summary:")
        logger.info(f"Total visualizations generated: {len(os.listdir(self.viz_dir))}")
        logger.info(f"Visualization directory: {self.viz_dir}")
        logger.info(f"Metrics file: {metrics_file}")

if __name__ == '__main__':
    unittest.main() 