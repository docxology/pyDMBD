#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Runner for DMBD Gaussian Blob Examples and Tests.

This script provides a unified interface for running various Gaussian Blob tests
and examples with Dynamic Markov Blanket Detection (DMBD). It supports both the
stabilized example implementations and the comprehensive integration tests.

The Gaussian Blob serves as an ideal test case for DMBD because:
1. It has clear ground truth with known internal, blanket, and external regions
2. It provides visual validation of the discovered Markov blanket structure
3. It tests the algorithm's ability to identify causal structure in dynamic systems

Usage:
    # Run the stabilized example with default settings
    python run_gaussian_blob_tests.py --mode example
    
    # Run comprehensive integration tests
    python run_gaussian_blob_tests.py --mode integration
    
    # Run stability tests
    python run_gaussian_blob_tests.py --mode stability
    
    # Run all tests with comprehensive reporting
    python run_gaussian_blob_tests.py --mode all --report
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"gaussian_blob_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("gaussian_blob_runner")

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def find_script(script_name, search_paths=None):
    """
    Find a script file in multiple possible locations.
    
    Args:
        script_name (str): Name of the script to find
        search_paths (list): Optional list of paths to search in addition to defaults
        
    Returns:
        Path or None: Path to the script if found, None otherwise
    """
    if search_paths is None:
        search_paths = []
        
    # Default search paths
    default_paths = [
        script_dir,
        os.path.join(script_dir, "examples"),
        os.path.join(script_dir, "tests"),
        os.path.join(os.path.dirname(script_dir), "examples"),
        os.path.join(os.path.dirname(script_dir), "tests")
    ]
    
    # Combine default and provided search paths
    all_paths = default_paths + search_paths
    
    # Search for the script
    for path in all_paths:
        script_path = os.path.join(path, script_name)
        if os.path.exists(script_path):
            return script_path
            
    # Try alternative names if not found
    alternatives = {
        "run_gaussian_blob.py": ["run_gaussian_blob_fixed.py", "dmbd_gaussian_blob_stabilized.py"],
        "run_gaussian_blob_fixed.py": ["run_gaussian_blob.py", "dmbd_gaussian_blob_stabilized.py"],
        "dmbd_gaussian_blob_stabilized.py": ["run_gaussian_blob_fixed.py", "run_gaussian_blob.py"]
    }
    
    if script_name in alternatives:
        for alt_name in alternatives[script_name]:
            for path in all_paths:
                alt_path = os.path.join(path, alt_name)
                if os.path.exists(alt_path):
                    logger.info(f"Found alternative script {alt_path} instead of {script_name}")
                    return alt_path
    
    return None

def run_example(args):
    """
    Run the stabilized Gaussian blob example with specified arguments.
    
    This runs the stand-alone example that demonstrates DMBD with the Gaussian Blob
    without the full testing infrastructure.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    logger.info("Running stabilized Gaussian blob example...")
    
    # Find the example script
    script_path = find_script("dmbd_gaussian_blob_stabilized.py")
    if not script_path:
        script_path = find_script("run_gaussian_blob_fixed.py")
        
    if not script_path:
        logger.error("Could not find the Gaussian Blob example script")
        return False
    
    # Construct command
    cmd = [sys.executable, script_path]
    
    # Add arguments
    if args.iterations:
        cmd.extend(["--iterations", str(args.iterations)])
    if args.reg:
        cmd.extend(["--reg", str(args.reg)])
    if args.grid_size:
        cmd.extend(["--grid-size", str(args.grid_size)])
    if args.time_steps:
        cmd.extend(["--time-steps", str(args.time_steps)])
    if args.feature_method:
        cmd.extend(["--feature-method", args.feature_method])
    if args.visualize:
        cmd.append("--visualize")
    
    # Create output directory
    output_dir = args.output_dir or "gaussian_blob_example_outputs"
    os.makedirs(output_dir, exist_ok=True)
    cmd.extend(["--output-dir", output_dir])
    
    # Run the command
    logger.info(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Example completed successfully")
        
        # Log the important parts of the output
        important_lines = [line for line in result.stdout.split('\n') if 
                         ('accuracy' in line.lower() or 
                          'dmbd' in line.lower() or
                          'success' in line.lower())]
        if important_lines:
            logger.info("Key output:")
            for line in important_lines[:10]:  # Limit to 10 lines
                logger.info(f"  {line}")
                
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Example failed with exit code {e.returncode}")
        if e.stdout:
            logger.error("Output:")
            logger.error(e.stdout)
        if e.stderr:
            logger.error("Error:")
            logger.error(e.stderr)
        return False

def run_integration_tests(args):
    """
    Run the Gaussian Blob integration tests with specified arguments.
    
    This runs the formal test suite that validates DMBD functionality with
    the Gaussian Blob example.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    logger.info("Running Gaussian Blob integration tests...")
    
    # Find the test script
    test_script = find_script("test_dmbd_gaussian_blob_integration.py")
    if not test_script:
        logger.error("Could not find the Gaussian Blob integration test script")
        return False
    
    # Construct command
    cmd = [sys.executable, "-m", "pytest", test_script, "-v"]
    
    # Add arguments
    if args.quick:
        cmd.append("-k quick")
        os.environ['DMBD_QUICK_TEST'] = '1'
    
    # Add timeout if provided
    if args.timeout:
        cmd.append(f"--timeout={args.timeout}")
    
    # Add seed if provided
    if args.seed:
        cmd.append(f"--randomly-seed={args.seed}")
    
    # Create output directory
    output_dir = args.output_dir or "gaussian_blob_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variable for output directory
    os.environ['DMBD_TEST_OUTPUT_DIR'] = output_dir
    
    # Run the command
    logger.info(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        
        # Save output to file
        output_file = os.path.join(output_dir, "integration_test_output.txt")
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nERRORS:\n")
                f.write(result.stderr)
        
        logger.info(f"Test {'succeeded' if success else 'failed'}")
        logger.info(f"Output saved to {output_file}")
        
        # Log the summary
        summary_lines = [line for line in result.stdout.split('\n') if 
                        ('PASSED' in line or 'FAILED' in line or 
                         'collected' in line or 'failed' in line)]
        if summary_lines:
            logger.info("Test Summary:")
            for line in summary_lines:
                logger.info(f"  {line}")
        
        return success
    except Exception as e:
        logger.error(f"Error running integration tests: {e}")
        return False

def run_stability_tests(args):
    """
    Run the Gaussian Blob numerical stability tests.
    
    This runs tests specifically focused on numerical stability issues
    that can arise with DMBD and the Gaussian Blob example.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    logger.info("Running DMBD Gaussian blob stability tests...")
    
    # Find the stability test script
    test_script = find_script("test_dmbd_gaussian_blob_stability.py")
    if not test_script:
        logger.error("Could not find the Gaussian Blob stability test script")
        return False
    
    # Create output directory
    output_dir = args.output_dir or "gaussian_blob_stability_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    cmd = [sys.executable, "-m", "pytest", test_script, "-v"]
    
    # Add specific test types if specified
    test_options = []
    if args.test_matrix_inversion:
        test_options.append("matrix_inversion")
    if args.test_regularization:
        test_options.append("regularization")
    if args.test_feature_methods:
        test_options.append("feature")
    if args.test_consistency:
        test_options.append("consistency")
    
    if test_options:
        test_expr = " or ".join(test_options)
        cmd.append(f"-k '{test_expr}'")
    
    # Add timeout if provided
    if args.timeout:
        cmd.append(f"--timeout={args.timeout}")
    
    # Add seed if provided
    if args.seed:
        cmd.append(f"--randomly-seed={args.seed}")
    
    # Set environment variables
    os.environ['DMBD_TEST_OUTPUT_DIR'] = output_dir
    if args.quick:
        os.environ['DMBD_QUICK_TEST'] = '1'
    
    # Run the command
    logger.info(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        
        # Save output to file
        output_file = os.path.join(output_dir, "stability_test_output.txt")
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nERRORS:\n")
                f.write(result.stderr)
        
        logger.info(f"Stability tests {'succeeded' if success else 'failed'}")
        logger.info(f"Output saved to {output_file}")
        
        # Log the summary
        summary_lines = [line for line in result.stdout.split('\n') if 
                        ('PASSED' in line or 'FAILED' in line or 
                         'collected' in line or 'failed' in line)]
        if summary_lines:
            logger.info("Test Summary:")
            for line in summary_lines:
                logger.info(f"  {line}")
        
        return success
    except Exception as e:
        logger.error(f"Error running stability tests: {e}")
        return False
    
def run_comprehensive_test_suite(args):
    """
    Run the full comprehensive test suite for Gaussian Blob with DMBD.
    
    This runs all available tests and examples, providing a comprehensive
    validation of DMBD functionality with the Gaussian Blob simulation.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Test results with success/failure status for each component
    """
    logger.info("Running comprehensive Gaussian Blob test suite...")
    
    # Create master output directory
    master_output_dir = args.output_dir or "gaussian_blob_comprehensive_outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join(master_output_dir, f"run_{timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    
    # Results dictionary
    results = {
        "timestamp": timestamp,
        "config": {
            "quick_mode": args.quick,
            "grid_size": args.grid_size,
            "time_steps": args.time_steps,
            "timeout": args.timeout,
            "seed": args.seed
        },
        "tests": {}
    }
    
    # 1. Run the example
    example_output_dir = os.path.join(master_output_dir, "example")
    os.makedirs(example_output_dir, exist_ok=True)
    
    example_args = argparse.Namespace(**vars(args))
    example_args.output_dir = example_output_dir
    
    logger.info("="*80)
    logger.info("STEP 1: Running Gaussian Blob Example")
    logger.info("="*80)
    
    start_time = time.time()
    example_success = run_example(example_args)
    example_duration = time.time() - start_time
    
    results["tests"]["example"] = {
        "success": example_success,
        "duration": example_duration,
        "output_dir": example_output_dir
    }
    
    # 2. Run integration tests
    integration_output_dir = os.path.join(master_output_dir, "integration")
    os.makedirs(integration_output_dir, exist_ok=True)
    
    integration_args = argparse.Namespace(**vars(args))
    integration_args.output_dir = integration_output_dir
    
    logger.info("\n"+"="*80)
    logger.info("STEP 2: Running Integration Tests")
    logger.info("="*80)
    
    start_time = time.time()
    integration_success = run_integration_tests(integration_args)
    integration_duration = time.time() - start_time
    
    results["tests"]["integration"] = {
        "success": integration_success,
        "duration": integration_duration,
        "output_dir": integration_output_dir
    }
    
    # 3. Run stability tests
    stability_output_dir = os.path.join(master_output_dir, "stability")
    os.makedirs(stability_output_dir, exist_ok=True)
    
    stability_args = argparse.Namespace(**vars(args))
    stability_args.output_dir = stability_output_dir
    stability_args.test_all = True
    
    logger.info("\n"+"="*80)
    logger.info("STEP 3: Running Stability Tests")
    logger.info("="*80)
    
    start_time = time.time()
    stability_success = run_stability_tests(stability_args)
    stability_duration = time.time() - start_time
    
    results["tests"]["stability"] = {
        "success": stability_success,
        "duration": stability_duration,
        "output_dir": stability_output_dir
    }
    
    # Calculate overall success
    overall_success = example_success and integration_success and stability_success
    results["overall_success"] = overall_success
    results["total_duration"] = example_duration + integration_duration + stability_duration
    
    # Create summary report
    logger.info("\n"+"="*80)
    logger.info("Test Suite Summary")
    logger.info("="*80)
    logger.info(f"Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    logger.info(f"Example Tests: {'SUCCESS' if example_success else 'FAILURE'} ({example_duration:.2f}s)")
    logger.info(f"Integration Tests: {'SUCCESS' if integration_success else 'FAILURE'} ({integration_duration:.2f}s)")
    logger.info(f"Stability Tests: {'SUCCESS' if stability_success else 'FAILURE'} ({stability_duration:.2f}s)")
    logger.info(f"Total Duration: {results['total_duration']:.2f}s")
    
    # Save results to JSON
    results_file = os.path.join(master_output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate HTML report if requested
    if args.report:
        try:
            # Generate HTML report
            html_report = generate_html_report(results, master_output_dir)
            logger.info(f"HTML report generated: {html_report}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
    
    # Return the results
    return results

def generate_html_report(results, output_dir):
    """
    Generate an HTML report from test results.
    
    Args:
        results (dict): Test results dictionary
        output_dir (str): Output directory for the report
        
    Returns:
        str: Path to the generated HTML report
    """
    report_file = os.path.join(output_dir, "test_report.html")
    
    # Generate a simple HTML report
    with open(report_file, 'w') as f:
        # Header
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Gaussian Blob DMBD Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .test-section {{ margin-top: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Gaussian Blob DMBD Test Report</h1>
    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Overall Result:</strong> <span class="{'success' if results['overall_success'] else 'failure'}">{'SUCCESS' if results['overall_success'] else 'FAILURE'}</span></p>
        <p><strong>Timestamp:</strong> {results['timestamp']}</p>
        <p><strong>Total Duration:</strong> {results['total_duration']:.2f} seconds</p>
        
        <h3>Configuration</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Quick Mode</td><td>{results['config']['quick_mode']}</td></tr>
            <tr><td>Grid Size</td><td>{results['config']['grid_size']}</td></tr>
            <tr><td>Time Steps</td><td>{results['config']['time_steps']}</td></tr>
            <tr><td>Timeout</td><td>{results['config']['timeout']}</td></tr>
            <tr><td>Seed</td><td>{results['config']['seed']}</td></tr>
        </table>
    </div>
""")
        
        # Test details
        for test_name, test_result in results["tests"].items():
            f.write(f"""
    <div class="test-section">
        <h2>{test_name.capitalize()} Tests</h2>
        <p><strong>Result:</strong> <span class="{'success' if test_result['success'] else 'failure'}">{'SUCCESS' if test_result['success'] else 'FAILURE'}</span></p>
        <p><strong>Duration:</strong> {test_result['duration']:.2f} seconds</p>
        <p><strong>Output Directory:</strong> {test_result['output_dir']}</p>
    </div>
""")
        
        # Footer
        f.write("""
    <div style="margin-top: 30px; text-align: center; color: #666;">
        <p>Generated by the Gaussian Blob DMBD Test Suite</p>
    </div>
</body>
</html>
""")
    
    return report_file

def main():
    """Parse command-line arguments and run the requested mode."""
    parser = argparse.ArgumentParser(description="Run DMBD Gaussian Blob examples and tests")
    
    # Mode selection
    parser.add_argument("--mode", choices=["example", "integration", "stability", "all"], default="example",
                       help="Run mode: 'example' to run the stabilized example, 'integration' for integration tests, 'stability' for stability tests, 'all' for the full suite")
    
    # Common options
    parser.add_argument("--quick", action="store_true", help="Run with reduced settings for faster testing")
    parser.add_argument("--output-dir", type=str, help="Directory to store outputs")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    
    # Example mode arguments
    example_group = parser.add_argument_group("Example Mode Options")
    example_group.add_argument("--iterations", type=int, help="Number of DMBD update iterations")
    example_group.add_argument("--reg", type=float, help="Regularization strength")
    example_group.add_argument("--grid-size", type=int, default=16, help="Grid size")
    example_group.add_argument("--time-steps", type=int, default=50, help="Number of time steps")
    example_group.add_argument("--feature-method", choices=["basic", "spatial", "roles"],
                             help="Feature extraction method")
    example_group.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    # Stability test mode arguments
    stability_group = parser.add_argument_group("Stability Test Mode Options")
    stability_group.add_argument("--test-all", action="store_true", help="Run all stability tests")
    stability_group.add_argument("--test-matrix-inversion", action="store_true", help="Run matrix inversion tests")
    stability_group.add_argument("--test-regularization", action="store_true", help="Run model regularization tests")
    stability_group.add_argument("--test-feature-methods", action="store_true", help="Run feature extraction method tests")
    stability_group.add_argument("--test-consistency", action="store_true", help="Run consistency tests")
    
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    
    # Run the requested mode
    try:
    if args.mode == "example":
        success = run_example(args)
        elif args.mode == "integration":
            success = run_integration_tests(args)
        elif args.mode == "stability":
            success = run_stability_tests(args)
        elif args.mode == "all":
            results = run_comprehensive_test_suite(args)
            success = results["overall_success"]
    else:
        logger.error(f"Unknown mode: {args.mode}")
            success = False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        success = False
    
    # Record and report elapsed time
    elapsed_time = datetime.now() - start_time
    logger.info(f"Total run time: {elapsed_time}")
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 