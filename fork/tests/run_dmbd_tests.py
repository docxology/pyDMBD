#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner script for DMBD tests.

This script runs all DMBD test suites and generates a comprehensive report
on the tensor dimensions and inference capabilities of the DMBD model.
"""

import os
import sys
import unittest
import logging
import json
import argparse
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import importlib
import traceback
import io
import contextlib

# Define output directories
TEST_OUTPUT_DIR = Path("fork/test_results")
TEST_REPORTS_DIR = TEST_OUTPUT_DIR / "reports"
TEST_LOGS_DIR = TEST_OUTPUT_DIR / "logs"

# Create output directories if they don't exist
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TEST_LOGS_DIR / 'dmbd_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dmbd_test_runner")

logger.info(f"Test outputs will be saved to: {TEST_OUTPUT_DIR}")
logger.info(f"Test reports will be saved to: {TEST_REPORTS_DIR}")
logger.info(f"Test logs will be saved to: {TEST_LOGS_DIR}")

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Add the tests directory to the path
tests_dir = str(Path(__file__).parent)
sys.path.insert(0, tests_dir)

# Import local modules
try:
    from fork.dmbd.dmbd import DMBD
    logger.info("Successfully imported DMBD module")
except ImportError:
    try:
        from dmbd.dmbd import DMBD
        logger.info("Successfully imported DMBD module from alternative path")
    except ImportError:
        logger.error("Failed to import DMBD module - tests may fail")

# Import test modules
try:
    from test_dmbd_basic import *
    from test_dmbd_dimensions import TestDMBDDimensions
    from test_dmbd_gaussian_blob_stability import TestDMBDGaussianBlobStability
    from test_dmbd_gaussian_blob_integration import TestDMBDGaussianBlobIntegration
    from test_dmbd_numerical_stability import TestDMBDNumericalStability
    from test_environment import *
    from test_examples import *
    logger.info("Successfully imported all test modules")
except ImportError as e:
    logger.error(f"Could not import test modules: {str(e)}")
    logger.error("Make sure they exist in the tests directory.")
    sys.exit(1)

# Initialize test modules dictionary with all test classes
test_modules = {
    'basic': 'test_dmbd_basic',
    'dimensions': TestDMBDDimensions,
    'gaussian_blob_stability': TestDMBDGaussianBlobStability,
    'gaussian_blob_integration': TestDMBDGaussianBlobIntegration,
    'numerical_stability': TestDMBDNumericalStability,
    'environment': 'test_environment',
    'examples': 'test_examples'
}

logger.info(f"Successfully imported {len(test_modules)} test modules")


def collect_tensor_metrics(model, data):
    """
    Collect metrics about tensor dimensions and operations in the DMBD model.
    
    Args:
        model: DMBD model instance
        data: Input data tensor
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "model_parameters": {},
        "tensor_counts": 0,
        "parameter_counts": 0,
        "gradients_supported": False,
        "matrix_shapes": {}
    }
    
    # Collect model parameter information
    if hasattr(model, 'obs_shape'):
        metrics["model_parameters"]["obs_shape"] = list(model.obs_shape)
    if hasattr(model, 'role_dims'):
        metrics["model_parameters"]["role_dims"] = model.role_dims
    if hasattr(model, 'hidden_dims'):
        metrics["model_parameters"]["hidden_dims"] = model.hidden_dims
    if hasattr(model, 'number_of_objects'):
        metrics["model_parameters"]["number_of_objects"] = model.number_of_objects
    
    # Count tensors and parameters
    def count_tensors(obj):
        count = {"tensors": 0, "parameters": 0, "total_elements": 0}
        if isinstance(obj, torch.Tensor):
            count["tensors"] += 1
            count["total_elements"] += obj.numel()
            if isinstance(obj, torch.nn.Parameter):
                count["parameters"] += 1
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                sub_count = count_tensors(item)
                count["tensors"] += sub_count["tensors"]
                count["parameters"] += sub_count["parameters"]
                count["total_elements"] += sub_count["total_elements"]
        elif hasattr(obj, '__dict__'):
            for key, val in obj.__dict__.items():
                if not key.startswith('_'):
                    sub_count = count_tensors(val)
                    count["tensors"] += sub_count["tensors"]
                    count["parameters"] += sub_count["parameters"]
                    count["total_elements"] += sub_count["total_elements"]
        return count
    
    tensor_counts = count_tensors(model)
    metrics["tensor_counts"] = tensor_counts["tensors"]
    metrics["parameter_counts"] = tensor_counts["parameters"]
    metrics["total_elements"] = tensor_counts["total_elements"]
    
    # Check matrix shapes
    if hasattr(model, 'A') and hasattr(model.A, 'data'):
        metrics["matrix_shapes"]["A"] = list(model.A.data.shape)
    if hasattr(model, 'C') and hasattr(model.C, 'data'):
        metrics["matrix_shapes"]["C"] = list(model.C.data.shape)
    
    # Check gradient support
    try:
        if data is not None:
            test_data = data.clone().detach().requires_grad_(True)
            # Try a simple update and check if gradients flow
            initial_count = torch.autograd._execution_engine.n_executed_nodes \
                if hasattr(torch.autograd, '_execution_engine') else 0
            
            with torch.autograd.set_detect_anomaly(True):
                success = model.update(
                    test_data, 
                    None, 
                    None,
                    iters=1,
                    lr=0.0001,
                    verbose=False
                )
                
                final_count = torch.autograd._execution_engine.n_executed_nodes \
                    if hasattr(torch.autograd, '_execution_engine') else 0
                op_count = final_count - initial_count
                
                metrics["gradients_supported"] = op_count > 0
                metrics["pytorch_op_count"] = op_count
    except Exception as e:
        logger.warning(f"Error checking gradient support: {str(e)}")
        metrics["gradients_supported"] = False
    
    return metrics


def run_tests_and_collect_metrics():
    """
    Run all DMBD tests and collect metrics about tensor dimensions and operations.
    
    Returns:
        dict: Dictionary of test results and metrics
    """
    # Create output directory for reports
    output_dir = Path("test_outputs/reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the test suites
    suite = unittest.TestSuite()
    
    # Add dimension tests
    dims_loader = unittest.TestLoader()
    dims_tests = dims_loader.loadTestsFromTestCase(TestDMBDDimensions)
    suite.addTests(dims_tests)
    
    # Add integration tests
    integration_loader = unittest.TestLoader()
    integration_tests = integration_loader.loadTestsFromTestCase(TestDMBDGaussianBlobIntegration)
    suite.addTests(integration_tests)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Collect results
    test_results = {
        "run_date": datetime.datetime.now().isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful()
    }
    
    # Collect metrics from a simple DMBD model
    model = DMBD(
        obs_shape=(1, 6),
        role_dims=[2, 2, 2],
        hidden_dims=[2, 2, 2],
        number_of_objects=1
    )
    
    # Create sample data
    sample_data = torch.rand((10, 1, 6))
    
    # Collect metrics
    model_metrics = collect_tensor_metrics(model, sample_data)
    
    # Combine results
    report = {
        "test_results": test_results,
        "model_metrics": model_metrics
    }
    
    # Save report to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"dmbd_test_report_{timestamp}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {report_file}")
    
    return report


def create_report_visualizations(report, output_dir=None):
    """
    Create visualizations from the test report.
    
    Args:
        report: Dictionary containing test results and metrics
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = Path("test_outputs/reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot test results
    test_results = report["test_results"]
    ax1 = axes[0, 0]
    ax1.bar(["Total", "Failures", "Errors", "Skipped"],
           [test_results["tests_run"], test_results["failures"],
            test_results["errors"], test_results["skipped"]])
    ax1.set_title("Test Results")
    ax1.set_ylabel("Count")
    
    # Plot tensor counts
    model_metrics = report["model_metrics"]
    ax2 = axes[0, 1]
    ax2.bar(["Tensors", "Parameters"],
           [model_metrics["tensor_counts"], model_metrics["parameter_counts"]])
    ax2.set_title("DMBD Model Tensor Counts")
    ax2.set_ylabel("Count")
    
    # Plot matrix shapes
    ax3 = axes[1, 0]
    matrix_shapes = model_metrics.get("matrix_shapes", {})
    for matrix, shape in matrix_shapes.items():
        ax3.bar(f"{matrix} dimensions", len(shape))
        ax3.text(f"{matrix} dimensions", len(shape) + 0.1, f"{shape}", ha='center')
    ax3.set_title("Matrix Dimensions")
    ax3.set_ylabel("Number of dimensions")
    
    # Plot role and hidden dimensions
    ax4 = axes[1, 1]
    role_dims = model_metrics.get("model_parameters", {}).get("role_dims", [])
    hidden_dims = model_metrics.get("model_parameters", {}).get("hidden_dims", [])
    
    if role_dims and hidden_dims:
        x = range(len(role_dims))
        width = 0.35
        ax4.bar([i - width/2 for i in x], role_dims, width, label='Role dims')
        ax4.bar([i + width/2 for i in x], hidden_dims, width, label='Hidden dims')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Internal', 'Blanket', 'External'])
        ax4.set_title("Role and Hidden Dimensions")
        ax4.set_ylabel("Dimension size")
        ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_file = output_dir / f"dmbd_test_summary_{timestamp}.png"
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    logger.info(f"Visualizations saved to {fig_file}")


class DMBDTestRunner:
    """Main test runner for DMBD tests"""
    
    def __init__(self):
        """Initialize the test runner."""
        self.results = {}  # Store test results
        self.test_counts = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        }
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = None
    
    def run_numerical_stability_test(self):
        """Run the numerical stability test for DMBD."""
        logger.info("Running DMBD numerical stability test...")
        
        if 'numerical_stability' not in test_modules:
            logger.error("Numerical stability test module not available. Skipping test.")
            self.results['numerical_stability'] = {
                'status': 'SKIPPED',
                'message': 'Test module not available',
                'output': '',
                'error': 'Module import failed'
            }
            self.test_counts['skipped'] += 1
            return
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # Create an instance of the numerical stability tester
                test_suite = unittest.TestSuite()
                loader = unittest.TestLoader()
                stability_tests = loader.loadTestsFromTestCase(test_modules['numerical_stability'].TestDMBDNumericalStability)
                test_suite.addTests(stability_tests)
                
                # Run the tests with a result collector
                result = unittest.TextTestRunner(verbosity=2).run(test_suite)
                
                # Check if all tests passed
                all_passed = result.wasSuccessful()
                tests_run = result.testsRun
                failure_count = len(result.failures)
                error_count = len(result.errors)
                
                # Log success/failure details
                if all_passed:
                    logger.info(f"Numerical stability test passed ({tests_run} tests)")
                    status = "PASSED"
                    message = f"All {tests_run} numerical stability tests passed"
                    self.test_counts['passed'] += 1
                else:
                    logger.warning(f"Numerical stability test failed: {failure_count} failures, {error_count} errors")
                    
                    # Create a detailed message
                    message = f"Numerical stability test: {failure_count} failures, {error_count} errors"
                    if result.failures:
                        message += f"\nFirst failure: {result.failures[0][1][:200]}..."
                
                self.test_counts['total'] += 1
            
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            
            # Check for errors in stderr
            if "Error" in stderr_output or "Exception" in stderr_output:
                status = "FAILED"
                message = "Errors detected during numerical stability test execution"
                self.test_counts['failed'] += 1
            else:
                if all_passed:
                    status = "PASSED"
                    message = "All numerical stability tests passed"
                    self.test_counts['passed'] += 1
                else:
                    status = "FAILED"
                    message = f"Some numerical stability tests failed. Passed {self.test_counts['passed']}/{self.test_counts['total']} tests."
                    self.test_counts['failed'] += 1
            
            self.test_counts['total'] += 1
            
        except Exception as e:
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue() + "\n" + traceback.format_exc()
            status = "ERROR"
            message = f"Exception during numerical stability test: {str(e)}"
            self.test_counts['errors'] += 1
            self.test_counts['total'] += 1
        
        self.results['numerical_stability'] = {
            'status': status,
            'message': message,
            'output': stdout_output,
            'error': stderr_output
        }
        
        logger.info(f"Numerical stability test {status}: {message}")
        return
    
    def run_minimal_test(self):
        """Run the minimal DMBD test."""
        logger.info("Running minimal DMBD test...")
        
        if 'minimal_test' not in test_modules:
            logger.error("Minimal test module not available. Skipping test.")
            self.results['minimal_test'] = {
                'status': 'SKIPPED',
                'message': 'Test module not available',
                'output': '',
                'error': 'Module import failed'
            }
            self.test_counts['skipped'] += 1
            return
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # Run the minimal test
                if hasattr(test_modules['minimal_test'], 'main'):
                    test_modules['minimal_test'].main()
                else:
                    # Try to run the test directly
                    test_modules['minimal_test']
            
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            
            # Check for errors in stderr
            if "Error" in stderr_output or "Exception" in stderr_output:
                status = "FAILED"
                message = "Errors detected during minimal test execution"
                self.test_counts['failed'] += 1
            else:
                status = "PASSED"
                message = "Minimal test completed successfully"
                self.test_counts['passed'] += 1
            
            self.test_counts['total'] += 1
            
        except Exception as e:
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue() + "\n" + traceback.format_exc()
            status = "FAILED"
            message = f"Exception during minimal test: {str(e)}"
            self.test_counts['errors'] += 1
            self.test_counts['total'] += 1
        
        self.results['minimal_test'] = {
            'status': status,
            'message': message,
            'output': stdout_output,
            'error': stderr_output
        }
        
        logger.info(f"Minimal DMBD test {status}: {message}")
        return
    
    def run_dimensions_test(self):
        """Run the DMBD dimensions test."""
        try:
            from test_dmbd_dimensions import TestDMBDDimensions
            logger.info("Successfully imported DMBD dimensions test")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestDMBDDimensions)
            
            # Run the test suite
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Store results
            self.results['dimensions_test'] = {
                'status': 'PASSED' if result.wasSuccessful() else 'FAILED',
                'message': 'Dimensions test completed successfully' if result.wasSuccessful() else 'Dimensions test failed',
                'errors': [str(error[1]) for error in result.errors],
                'failures': [str(failure[1]) for failure in result.failures]
            }
            
            # Log results
            if result.wasSuccessful():
                logger.info("Dimensions test PASSED: All tests completed successfully")
            else:
                logger.warning(f"Dimensions test FAILED: {len(result.errors)} errors, {len(result.failures)} failures")
            
        except ImportError as e:
            logger.warning("Failed to import dimensions test module")
            self.results['dimensions_test'] = {
                'status': 'SKIPPED',
                'message': 'Test module not available',
                'errors': [str(e)],
                'failures': []
            }
        except Exception as e:
            logger.error(f"Error running dimensions test: {str(e)}")
            self.results['dimensions_test'] = {
                'status': 'ERROR',
                'message': f'Error running dimensions test: {str(e)}',
                'errors': [str(e)],
                'failures': []
            }
    
    def run_gaussian_blob_test(self):
        """Run the DMBD Gaussian blob integration test."""
        try:
            # Add examples directory to Python path
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
            if examples_dir not in sys.path:
                sys.path.insert(0, examples_dir)
            
            from test_dmbd_gaussian_blob_integration import TestDMBDGaussianBlobIntegration
            logger.info("Successfully imported Gaussian blob integration test")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestDMBDGaussianBlobIntegration)
            
            # Run the test suite
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Store results
            self.results['gaussian_blob_test'] = {
                'status': 'PASSED' if result.wasSuccessful() else 'FAILED',
                'message': 'Gaussian blob test completed successfully' if result.wasSuccessful() else 'Gaussian blob test failed',
                'errors': [str(error[1]) for error in result.errors],
                'failures': [str(failure[1]) for failure in result.failures]
            }
            
            # Log results
            if result.wasSuccessful():
                logger.info("Gaussian blob test PASSED: All tests completed successfully")
            else:
                logger.warning(f"Gaussian blob test FAILED: {len(result.errors)} errors, {len(result.failures)} failures")
            
        except ImportError as e:
            logger.warning("Failed to import Gaussian blob test module")
            self.results['gaussian_blob_test'] = {
                'status': 'SKIPPED',
                'message': 'Test module not available. Please ensure the GaussianBlob.py file exists in the examples directory.',
                'errors': [str(e)],
                'failures': []
            }
        except Exception as e:
            logger.error(f"Error running Gaussian blob test: {str(e)}")
            self.results['gaussian_blob_test'] = {
                'status': 'ERROR',
                'message': f'Error running Gaussian blob test: {str(e)}',
                'errors': [str(e)],
                'failures': []
            }
    
    def run_torch_functionality_test(self):
        """Run the PyTorch functionality test."""
        logger.info("Running PyTorch functionality test...")
        
        try:
            from test_torch_functionality import TestTorchFunctionality
            logger.info("Successfully imported PyTorch functionality test")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestTorchFunctionality)
            
            # Run the test suite
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Update test counts
            self.test_counts['total'] += result.testsRun
            self.test_counts['passed'] += result.testsRun - len(result.failures) - len(result.errors)
            self.test_counts['failed'] += len(result.failures)
            self.test_counts['errors'] += len(result.errors)
            
            # Store results
            self.results['torch_functionality'] = {
                'status': 'PASSED' if result.wasSuccessful() else 'FAILED',
                'message': 'PyTorch functionality test completed successfully' if result.wasSuccessful() else 'PyTorch functionality test failed',
                'errors': [str(error[1]) for error in result.errors],
                'failures': [str(failure[1]) for failure in result.failures]
            }
            
            # Log results
            if result.wasSuccessful():
                logger.info("PyTorch functionality test PASSED: All tests completed successfully")
            else:
                logger.warning(f"PyTorch functionality test FAILED: {len(result.errors)} errors, {len(result.failures)} failures")
            
        except ImportError as e:
            logger.warning("Failed to import PyTorch functionality test module")
            self.results['torch_functionality'] = {
                'status': 'SKIPPED',
                'message': 'Test module not available',
                'errors': [str(e)],
                'failures': []
            }
            self.test_counts['skipped'] += 1
            self.test_counts['total'] += 1
        except Exception as e:
            logger.error(f"Error running PyTorch functionality test: {str(e)}")
            self.results['torch_functionality'] = {
                'status': 'ERROR',
                'message': f'Error running PyTorch functionality test: {str(e)}',
                'errors': [str(e)],
                'failures': []
            }
            self.test_counts['errors'] += 1
            self.test_counts['total'] += 1
    
    def run_all_tests(self):
        """Run all available DMBD tests."""
        logger.info("Running all DMBD tests...")
        
        # Run each test if the module is available
        test_configs = [
            ('basic', 'test_dmbd_basic'),
            ('numerical_stability', TestDMBDNumericalStability),
            ('dimensions', TestDMBDDimensions),
            ('gaussian_blob_stability', TestDMBDGaussianBlobStability),
            ('gaussian_blob_integration', TestDMBDGaussianBlobIntegration),
            ('environment', 'test_environment'),
            ('examples', 'test_examples')
        ]
        
        for test_name, test_class in test_configs:
            if test_name in test_modules:
                logger.info(f"Running {test_name} tests...")
                try:
                    if isinstance(test_class, str):
                        # For modules that use pytest
                        import pytest
                        # Redirect pytest output to test_results directory
                        test_output_dir = TEST_OUTPUT_DIR / test_name
                        test_output_dir.mkdir(exist_ok=True)
                        
                        test_result = pytest.main([
                            '-v',
                            f'fork/tests/{test_class}.py',
                            f'--basetemp={test_output_dir}',
                            f'--junitxml={test_output_dir}/test_results.xml'
                        ])
                        success = test_result == 0
                    else:
                        # For unittest-based modules
                        # Create test-specific output directory
                        test_output_dir = TEST_OUTPUT_DIR / test_name
                        test_output_dir.mkdir(exist_ok=True)
                        
                        # Set up test environment
                        if hasattr(test_class, 'output_dir'):
                            test_class.output_dir = test_output_dir
                        
                        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                        with open(test_output_dir / 'test_output.log', 'w') as f:
                            result = unittest.TextTestRunner(
                                stream=f,
                                verbosity=2
                            ).run(suite)
                        success = result.wasSuccessful()
                    
                    self.results[test_name] = {
                        'status': 'PASSED' if success else 'FAILED',
                        'message': f'{test_name} tests completed successfully' if success else f'{test_name} tests failed',
                        'output_dir': str(test_output_dir)
                    }
                    
                    if success:
                        self.test_counts['passed'] += 1
                    else:
                        self.test_counts['failed'] += 1
                    
                except ImportError as e:
                    logger.warning(f"Skipping {test_name} tests due to missing dependencies: {str(e)}")
                    self.results[test_name] = {
                        'status': 'SKIPPED',
                        'message': f'Missing dependencies for {test_name} tests',
                        'error': str(e)
                    }
                    self.test_counts['skipped'] += 1
                except Exception as e:
                    logger.error(f"Error running {test_name} tests: {str(e)}")
                    self.results[test_name] = {
                        'status': 'ERROR',
                        'message': f'Error running {test_name} tests',
                        'error': str(e)
                    }
                    self.test_counts['errors'] += 1
            else:
                logger.warning(f"{test_name} test module not available. Skipping.")
                self.results[test_name] = {
                    'status': 'SKIPPED',
                    'message': 'Test module not available',
                    'output': '',
                    'error': 'Module import failed'
                }
                self.test_counts['skipped'] += 1
            
            self.test_counts['total'] += 1
        
        # Run PyTorch functionality test
        self.run_torch_functionality_test()
        
        logger.info(f"All tests completed. "
                    f"Total: {self.test_counts['total']}, "
                    f"Passed: {self.test_counts['passed']}, "
                    f"Failed: {self.test_counts['failed']}, "
                    f"Errors: {self.test_counts['errors']}, "
                    f"Skipped: {self.test_counts['skipped']}")
        return
    
    def generate_report(self, output_dir=None):
        """Generate a comprehensive test report"""
        logger.info("Generating comprehensive test report...")
        
        # Use default output directory if none specified
        if output_dir is None:
            output_dir = TEST_REPORTS_DIR
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"dmbd_test_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DMBD TEST REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("TEST SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total tests: {self.test_counts['total']}\n")
            f.write(f"Passed: {self.test_counts['passed']}\n")
            f.write(f"Failed: {self.test_counts['failed']}\n")
            f.write(f"Errors: {self.test_counts['errors']}\n")
            f.write(f"Skipped: {self.test_counts['skipped']}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")
            
            for test_name, result in self.results.items():
                if result is None:
                    continue
                    
                f.write(f"Test: {test_name}\n")
                f.write(f"Status: {result.get('status', 'UNKNOWN')}\n")
                f.write(f"Message: {result.get('message', 'No message')}\n")
                
                if 'output_dir' in result:
                    f.write(f"Output directory: {result['output_dir']}\n")
                
                if 'output' in result and result['output']:
                    f.write("\nOutput:\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['output'][:1000])  # Limit output to 1000 chars
                    if len(result['output']) > 1000:
                        f.write("\n... (output truncated) ...\n")
                
                if 'error' in result and result['error']:
                    f.write("\nErrors:\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['error'][:1000])  # Limit error to 1000 chars
                    if len(result['error']) > 1000:
                        f.write("\n... (error output truncated) ...\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            if self.test_counts['failed'] > 0 or self.test_counts['errors'] > 0:
                f.write("The following issues were detected:\n\n")
                
                for test_name, result in self.results.items():
                    if result.get('status') in ['FAILED', 'ERROR']:
                        f.write(f"- {test_name}: {result.get('message', 'Unknown issue')}\n")
                        if 'error' in result:
                            f.write(f"  Error: {result['error']}\n")
                
                f.write("\nRecommended actions:\n")
                f.write("1. Check the test output directories for detailed logs\n")
                f.write("2. Review any missing dependencies\n")
                f.write("3. Fix failing tests before proceeding\n")
            else:
                f.write("All tests passed successfully! No issues detected.\n")
        
        # Generate graphical summary if matplotlib is available
        try:
            self._generate_graphical_summary(output_dir)
        except Exception as e:
            logger.warning(f"Could not generate graphical summary: {str(e)}")
        
        logger.info(f"Report generated at {report_path}")
        self.report_path = report_path
        return report_path
    
    def _generate_graphical_summary(self, output_dir):
        """Generate a graphical summary of test results"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Pie chart of overall results
            labels = ['Passed', 'Failed', 'Errors', 'Skipped']
            sizes = [
                self.test_counts['passed'],
                self.test_counts['failed'],
                self.test_counts['errors'],
                self.test_counts['skipped']
            ]
            colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E']
            explode = (0.1, 0.1, 0.1, 0.1)  # explode all slices
            
            # Only include non-zero values in the pie chart
            non_zero_labels = []
            non_zero_sizes = []
            non_zero_colors = []
            non_zero_explode = []
            
            for i, size in enumerate(sizes):
                if size > 0:
                    non_zero_labels.append(labels[i])
                    non_zero_sizes.append(size)
                    non_zero_colors.append(colors[i])
                    non_zero_explode.append(explode[i])
            
            if sum(non_zero_sizes) > 0:  # Only create pie chart if we have data
                ax1.pie(
                    non_zero_sizes, 
                    explode=non_zero_explode, 
                    labels=non_zero_labels, 
                    colors=non_zero_colors,
                    autopct='%1.1f%%', 
                    shadow=True, 
                    startangle=90
                )
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax1.set_title('Test Results Summary')
            else:
                ax1.text(0.5, 0.5, "No tests run", ha='center', va='center', fontsize=14)
                ax1.axis('off')
            
            # Bar chart of individual test results
            test_names = []
            test_statuses = []
            
            for test_name, result in self.results.items():
                if result is not None:
                    test_names.append(test_name)
                    status = result.get('status', 'UNKNOWN')
                    test_statuses.append(status)
            
            if test_names:  # Only create bar chart if we have data
                # Map statuses to colors
                status_colors = {
                    'PASSED': '#4CAF50',  # Green
                    'FAILED': '#F44336',  # Red
                    'ERROR': '#FF9800',   # Orange
                    'SKIPPED': '#9E9E9E', # Gray
                    'UNKNOWN': '#2196F3'  # Blue
                }
                
                # Create bar colors based on status
                bar_colors = [status_colors.get(status, '#2196F3') for status in test_statuses]
                
                # Create the bar chart
                y_pos = range(len(test_names))
                ax2.barh(y_pos, [1] * len(test_names), color=bar_colors)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(test_names)
                ax2.set_xlabel('Status')
                ax2.set_title('Individual Test Results')
                
                # Add a legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=color, label=status)
                    for status, color in status_colors.items()
                    if status in test_statuses
                ]
                ax2.legend(handles=legend_elements, loc='upper right')
                
                # Remove x-axis ticks as they're not meaningful
                ax2.set_xticks([])
            else:
                ax2.text(0.5, 0.5, "No individual test data", ha='center', va='center', fontsize=14)
                ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dmbd_test_summary_{self.timestamp}.png"))
            plt.close()
            
            logger.info("Generated graphical summary of test results")
        except Exception as e:
            logger.warning(f"Failed to generate graphical summary: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DMBD tests")
    parser.add_argument("--test", choices=["all", "stability", "minimal", "dimensions", "gaussian", "torch_functionality"], 
                        default="all", help="Specify which test to run")
    parser.add_argument("--generate-report", action="store_true", 
                        help="Generate a detailed report of test results")
    parser.add_argument("--report-dir", default=None,
                        help="Directory to store test reports (defaults to tests/test_outputs/reports)")
    args = parser.parse_args()
    
    # Create test runner
    runner = DMBDTestRunner()
    
    # Run specified test
    if args.test == "all":
        runner.run_all_tests()
    elif args.test == "stability":
        runner.run_numerical_stability_test()
    elif args.test == "minimal":
        runner.run_minimal_test()
    elif args.test == "dimensions":
        runner.run_dimensions_test()
    elif args.test == "gaussian":
        runner.run_gaussian_blob_test()
    elif args.test == "torch_functionality":
        runner.run_torch_functionality_test()
    
    # Generate report if requested
    if args.generate_report:
        report_path = runner.generate_report(args.report_dir)
        if report_path:
            logger.info(f"Test report generated at: {report_path}")
            logger.info(f"Test logs are available at: {TEST_LOGS_DIR}/dmbd_test.log")
    
    # Return exit code based on test results
    success = runner.test_counts['failed'] == 0 and runner.test_counts['errors'] == 0
    sys.exit(0 if success else 1) 