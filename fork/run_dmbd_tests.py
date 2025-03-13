#!/usr/bin/env python3
"""
Run the DMBD test suite with optimized settings for memory management and visualization.

This script executes the tests specifically focused on Dynamic Markov Blanket Detection
algorithms with appropriate settings to prevent memory issues and ensure proper visualization.
Tests will continue to run even if some fail, providing comprehensive results.
"""

import os
import sys
import argparse
import subprocess
import time
import glob
from pathlib import Path
import traceback

def run_test(test_file, args, output_dir):
    """Run a single test file and return success/failure."""
    print(f"\n{'='*80}")
    print(f"Running test: {test_file}")
    print(f"{'='*80}")
    
    test_name = os.path.basename(test_file).replace(".py", "")
    
    # Create a specific output directory for this test's logs
    test_output_dir = output_dir / test_name
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create log file
    log_file = test_output_dir / f"{test_name}_log.txt"
    
    # Build pytest command for this specific test
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",  # Verbose output
        f"--timeout={args.timeout}",  # Set timeout
        "--showlocals",  # Show local variables on failure
        "--durations=10",  # Show the 10 slowest tests
        f"--randomly-seed={args.seed}",  # Set random seed for reproducibility
    ]
    
    if args.quick:
        # Set environment variable for quick mode
        os.environ['DMBD_QUICK_TEST'] = '1'
    
    # Execute the test
    start_time = time.time()
    
    try:
        # Run the test and capture output
        with open(log_file, 'w') as log:
            log.write(f"Running command: {' '.join(cmd)}\n\n")
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            log.write(process.stdout)
            
            # Print to console as well
            print(process.stdout)
    
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if the test passed
        success = process.returncode == 0
        status = "PASSED" if success else "FAILED"
        
        # Write summary to log file
        with open(log_file, 'a') as log:
            log.write(f"\n\n{'='*50}\n")
            log.write(f"Test {test_name} {status} in {duration:.2f} seconds\n")
            log.write(f"Exit code: {process.returncode}\n")
            
        print(f"\nTest {test_name} {status} in {duration:.2f} seconds")
        
        return {
            'test': test_name,
            'success': success,
            'duration': duration,
            'exit_code': process.returncode
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Error running test {test_name}: {e}")
        traceback.print_exc()
        
        # Write error to log file
        with open(log_file, 'a') as log:
            log.write(f"\n\nERROR running test {test_name}: {e}\n")
            log.write(traceback.format_exc())
            log.write(f"\nTest duration: {duration:.2f} seconds\n")
        
        return {
            'test': test_name,
            'success': False,
            'duration': duration,
            'exit_code': -1,
            'error': str(e)
        }

def run_gaussian_blob_tests(args, output_dir):
    """
    Run comprehensive tests on the Gaussian Blob example with various configurations.
    
    This function tests the Gaussian Blob example with different grid sizes, time steps, 
    and feature extraction methods to validate DMBD's ability to detect Markov blankets
    in this simulated environment.
    
    Args:
        args: Command line arguments
        output_dir: Base output directory for test results
        
    Returns:
        dict: Test results with success/failure status
    """
    print(f"\n{'='*80}")
    print(f"Running Gaussian Blob Tests")
    print(f"{'='*80}")
    
    # Create specific output directory for Gaussian Blob tests
    gb_output_dir = output_dir / "gaussian_blob_tests"
    os.makedirs(gb_output_dir, exist_ok=True)
    
    # Log file for Gaussian Blob tests
    log_file = gb_output_dir / "gaussian_blob_tests_log.txt"
    
    with open(log_file, 'w') as log:
        log.write(f"Gaussian Blob Tests\n")
        log.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Define test configurations to run
    test_configs = []
    
    # If quick mode, use minimal configurations
    if args.quick:
        test_configs = [
            {"grid_size": 8, "time_steps": 20, "feature_method": "basic"}
        ]
    else:
        # Full test suite with various configurations
        test_configs = [
            # Basic configuration with small grid
            {"grid_size": 8, "time_steps": 30, "feature_method": "basic"},
            
            # Medium grid with spatial features
            {"grid_size": 12, "time_steps": 40, "feature_method": "spatial"},
            
            # Larger grid with role-based features
            {"grid_size": 16, "time_steps": 50, "feature_method": "roles"}
        ]
    
    # Results for each configuration
    config_results = []
    
    # Run tests for each configuration
    for i, config in enumerate(test_configs):
        print(f"\nRunning Gaussian Blob test configuration {i+1}/{len(test_configs)}:")
        print(f"  Grid Size: {config['grid_size']}")
        print(f"  Time Steps: {config['time_steps']}")
        print(f"  Feature Method: {config['feature_method']}")
        
        # Create config-specific output directory
        config_dir = gb_output_dir / f"config_{i+1}"
        os.makedirs(config_dir, exist_ok=True)
        
        # Path to the run_gaussian_blob_fixed.py script
        # (using fixed version which has better stability)
        gb_script = Path(__file__).parent / "run_gaussian_blob_fixed.py"
        
        if not gb_script.exists():
            # Fall back to the original script if fixed version doesn't exist
            gb_script = Path(__file__).parent / "run_gaussian_blob.py"
            
            if not gb_script.exists():
                # Check if it's in the examples directory
                gb_script = Path(__file__).parent / "examples" / "dmbd_gaussian_blob_stabilized.py"
                
                if not gb_script.exists():
                    print(f"Error: Could not find Gaussian Blob script")
                    with open(log_file, 'a') as log:
                        log.write(f"Error: Could not find Gaussian Blob script\n")
                    
                    return {
                        'test': 'gaussian_blob',
                        'success': False,
                        'duration': 0,
                        'exit_code': -1,
                        'error': "Could not find Gaussian Blob script"
                    }
        
        # Build command for running the Gaussian Blob test
        cmd = [
            sys.executable, 
            str(gb_script),
            f"--grid-size={config['grid_size']}",
            f"--time-steps={config['time_steps']}",
            f"--output-dir={config_dir}",
            f"--seed={args.seed}",
            "--convergence-attempts=3"  # Limit attempts for testing
        ]
        
        # Add feature method if the script supports it
        if "feature_method" in config:
            cmd.append(f"--feature-method={config['feature_method']}")
        
        # Log the command
        with open(log_file, 'a') as log:
            log.write(f"\nConfiguration {i+1}:\n")
            log.write(f"Command: {' '.join(cmd)}\n")
        
        # Execute the test
        start_time = time.time()
        
        try:
            # Run the Gaussian Blob script
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Log output
            with open(log_file, 'a') as log:
                log.write(f"\nOutput:\n")
                log.write(process.stdout)
            
            # Print summary to console
            print(f"\nConfiguration {i+1} output summary:")
            output_lines = process.stdout.split('\n')
            # Print at most 10 important lines
            important_lines = [line for line in output_lines if 
                              ('accuracy' in line.lower() or 
                               'error' in line.lower() or 
                               'success' in line.lower() or
                               'dmbd' in line.lower())][:10]
            for line in important_lines:
                print(f"  {line}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Check for success
            success = process.returncode == 0
            config_results.append({
                'config': config,
                'success': success,
                'duration': duration,
                'exit_code': process.returncode
            })
            
            # Log result
            with open(log_file, 'a') as log:
                log.write(f"\nResult: {'SUCCESS' if success else 'FAILURE'}\n")
                log.write(f"Duration: {duration:.2f} seconds\n")
                log.write(f"Exit code: {process.returncode}\n")
                log.write(f"{'-'*50}\n")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Error running Gaussian Blob test configuration {i+1}: {e}")
            traceback.print_exc()
            
            # Log error
            with open(log_file, 'a') as log:
                log.write(f"\nERROR running configuration {i+1}: {e}\n")
                log.write(traceback.format_exc())
                log.write(f"\nDuration: {duration:.2f} seconds\n")
                log.write(f"{'-'*50}\n")
            
            config_results.append({
                'config': config,
                'success': False,
                'duration': duration,
                'exit_code': -1,
                'error': str(e)
            })
    
    # Check if any configuration succeeded
    overall_success = any(result['success'] for result in config_results)
    
    # Summarize results
    with open(log_file, 'a') as log:
        log.write(f"\nGaussian Blob Tests Summary:\n")
        log.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Configurations run: {len(config_results)}\n")
        log.write(f"Successful configurations: {sum(1 for r in config_results if r['success'])}\n")
        log.write(f"Failed configurations: {sum(1 for r in config_results if not r['success'])}\n")
        log.write(f"Overall result: {'SUCCESS' if overall_success else 'FAILURE'}\n\n")
        
        log.write(f"Configuration details:\n")
        for i, result in enumerate(config_results):
            log.write(f"  Config {i+1}: ")
            log.write(f"Grid={result['config']['grid_size']}, ")
            log.write(f"Steps={result['config']['time_steps']}, ")
            log.write(f"Method={result['config']['feature_method']}\n")
            log.write(f"    Result: {'SUCCESS' if result['success'] else 'FAILURE'}\n")
            log.write(f"    Duration: {result['duration']:.2f} seconds\n")
            if 'error' in result:
                log.write(f"    Error: {result['error']}\n")
    
    # Print summary to console
    print(f"\nGaussian Blob Tests Summary:")
    print(f"  Configurations run: {len(config_results)}")
    print(f"  Successful configurations: {sum(1 for r in config_results if r['success'])}")
    print(f"  Failed configurations: {sum(1 for r in config_results if not r['success'])}")
    print(f"  Overall result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    # Return overall result
    return {
        'test': 'gaussian_blob',
        'success': overall_success,
        'duration': sum(r['duration'] for r in config_results),
        'configurations': len(config_results),
        'successful_configs': sum(1 for r in config_results if r['success']),
        'config_results': config_results
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DMBD test suite')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='dmbd_outputs',
                      help='Directory to store outputs')
    parser.add_argument('--timeout', type=int, default=1200,
                      help='Timeout in seconds per test')
    parser.add_argument('--memory-limit', type=int, default=8192,
                      help='Memory limit in MB per test')
    parser.add_argument('--test-pattern', type=str, default='test_*dmbd*.py',
                      help='Pattern to match test files')
    parser.add_argument('--visualize-only', action='store_true',
                      help='Only run tests focused on visualization')
    parser.add_argument('--quick', action='store_true',
                      help='Run with reduced iterations and faster settings')
    parser.add_argument('--gaussian-blob-only', action='store_true',
                      help='Only run the Gaussian Blob tests')
    args = parser.parse_args()
    
    # Set environment variables to optimize memory usage
    os.environ['PYTHONMALLOC'] = 'debug'  # Enable Python memory debugging
    os.environ['PYTHONFAULTHANDLER'] = '1'  # Dump traceback on segfault
    os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads to reduce memory
    os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
    os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit numexpr threads
    
    # Set memory limit for tests if ulimit is available
    try:
        import resource
        # Set soft limit to args.memory_limit MB
        memory_limit_bytes = args.memory_limit * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        print(f"Set memory limit to {args.memory_limit} MB")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Could not set memory limit: {e}")
    
    # Set matplotlib to non-interactive backend to prevent display issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_dir}")
    
    # Run Gaussian Blob tests specifically if requested
    if args.gaussian_blob_only:
        print("Running only Gaussian Blob tests")
        results = [run_gaussian_blob_tests(args, output_dir)]
    else:
        # Find all test files matching the pattern
        test_dir = Path(__file__).parent / "tests"
        if args.test_pattern.endswith('.py'):
            pattern = args.test_pattern
        else:
            pattern = f"{args.test_pattern}.py" if not args.test_pattern.endswith('.py') else args.test_pattern
            
        test_files = glob.glob(str(test_dir / pattern))
        
        # Special case: if we didn't find any files matching the specific pattern,
        # but we have a dmbd pattern, try running the specialized DMBD test files
        if not test_files and 'dmbd' in args.test_pattern:
            test_files = [
                str(test_dir / "test_lorenz_dmbd.py"),
                str(test_dir / "test_newtons_cradle_dmbd.py"),
                str(test_dir / "test_dmbd_gaussian_blob_integration.py"),
                str(test_dir / "test_dmbd_dimensions.py"),
                str(test_dir / "test_dmbd_numerical_stability.py")
            ]
            # Filter to only existing files
            test_files = [f for f in test_files if os.path.exists(f)]
            
            # If still no files, try the main test_examples.py which has DMBD tests
            if not test_files:
                test_files = [str(test_dir / "test_examples.py")]
        
        if not test_files:
            print(f"No test files found matching pattern: {pattern}")
            return 1
        
        print(f"Found {len(test_files)} test files to run:")
        for test_file in test_files:
            print(f"  - {os.path.basename(test_file)}")
        
        # Run each test file individually to prevent one failure from stopping everything
        results = []
        for test_file in test_files:
            result = run_test(test_file, args, output_dir)
            results.append(result)
        
        # Run Gaussian Blob tests in addition to standard tests
        if not args.visualize_only:
            gaussian_blob_result = run_gaussian_blob_tests(args, output_dir)
            results.append(gaussian_blob_result)
    
    # Create a summary report
    summary_file = output_dir / "dmbd_test_summary.txt"
    with open(summary_file, "w") as f:
        f.write("DMBD Test Suite Summary\n")
        f.write("======================\n\n")
        f.write(f"Run completed on: {time.ctime()}\n")
        
        # Count successes and failures
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes
        
        f.write(f"Tests run: {len(results)}\n")
        f.write(f"Successes: {successes}\n")
        f.write(f"Failures: {failures}\n\n")
        
        # Environment info
        f.write("Environment:\n")
        f.write(f"  PYTHONMALLOC={os.environ.get('PYTHONMALLOC', 'default')}\n")
        f.write(f"  PYTHONFAULTHANDLER={os.environ.get('PYTHONFAULTHANDLER', '0')}\n")
        f.write(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}\n")
        f.write(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'not set')}\n")
        f.write(f"  Memory limit: {args.memory_limit} MB\n")
        f.write(f"  Timeout: {args.timeout} seconds\n\n")
        
        # Individual test results
        f.write("Test Results:\n")
        f.write("------------\n\n")
        
        for result in results:
            # Special handling for Gaussian Blob test which has multiple configurations
            if 'configurations' in result:
                status = "✓ PASSED" if result['success'] else "✗ FAILED"
                f.write(f"{status} - {result['test']} ({result['duration']:.2f}s)\n")
                f.write(f"  Configurations run: {result['configurations']}\n")
                f.write(f"  Successful configurations: {result['successful_configs']}\n")
                
                # Add details for each configuration
                for i, config_result in enumerate(result['config_results']):
                    config_status = "✓" if config_result['success'] else "✗"
                    f.write(f"  {config_status} Config {i+1}: Grid={config_result['config']['grid_size']}, ")
                    f.write(f"Steps={config_result['config']['time_steps']}, ")
                    f.write(f"Method={config_result['config']['feature_method']}\n")
                    if 'error' in config_result:
                        f.write(f"    Error: {config_result['error']}\n")
                f.write("\n")
            else:
                # Standard test result
                status = "✓ PASSED" if result['success'] else "✗ FAILED"
                f.write(f"{status} - {result['test']} ({result['duration']:.2f}s)\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
    
    print(f"\nTest Summary:")
    print(f"  Tests run: {len(results)}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"Summary written to {summary_file}")
    
    # Return success if all tests passed, failure otherwise
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 