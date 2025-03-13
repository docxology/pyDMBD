#!/usr/bin/env python3
"""
Quick script to generate Dynamic Markov Blanket Detection visualizations.

This script runs the necessary tests to generate visualizations of the Dynamic Markov
Blanket Detection algorithm applied to various example systems, handling failures
gracefully to ensure all possible visualizations are generated.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    print("=" * 80)
    print("Generating Dynamic Markov Blanket Detection (DMBD) Visualizations")
    print("=" * 80)
    
    # Create necessary output directories
    base_dir = Path(__file__).parent
    output_dir = base_dir / "dmbd_visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of example systems to generate visualizations for
    example_systems = [
        "lorenz",
        "newtons_cradle",
        "flame",
        "forager"
    ]
    
    # Set environment variables for better memory management
    os.environ['PYTHONMALLOC'] = 'debug'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['DMBD_QUICK_TEST'] = '1'  # Use quicker settings
    os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive matplotlib
    
    # Track results
    results = []
    
    # Try to run each example independently
    for system in example_systems:
        print(f"\nGenerating DMBD visualization for {system}...")
        
        # Determine which test to run based on system
        if system == "lorenz":
            test_file = base_dir / "tests" / "test_lorenz_dmbd.py"
            if not test_file.exists():
                # Fall back to general test
                test_file = base_dir / "tests" / "test_examples.py::test_lorenz_dmbd_inference"
                if not os.path.exists(test_file.parent):
                    test_file = base_dir / "tests" / "test_examples.py"
        elif system == "newtons_cradle":
            test_file = base_dir / "tests" / "test_newtons_cradle_dmbd.py"
            if not test_file.exists():
                # Fall back to general test
                test_file = base_dir / "tests" / "test_examples.py::test_newtons_cradle_dmbd_analysis"
                if not os.path.exists(test_file.parent):
                    test_file = base_dir / "tests" / "test_examples.py"
        else:
            # Use the general test file for other systems
            test_file = base_dir / "tests" / "test_examples.py::test_{}_example".format(system)
            if not os.path.exists(test_file.parent):
                print(f"  Warning: Test file not found for {system}")
                continue
        
        # Build command
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_file),
            "-v",
            "--timeout=600",
            "--capture=no"  # Show output in real-time
        ]
        
        # Execute the test
        start_time = time.time()
        try:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            # Ensure output is copied even if the test failed
            system_output_dir = base_dir / "dmbd_outputs" / system
            dest_dir = output_dir / system
            
            if os.path.exists(system_output_dir):
                # Copy files to visualization output directory
                os.makedirs(dest_dir, exist_ok=True)
                for filepath in system_output_dir.glob('*.png'):
                    try:
                        import shutil
                        shutil.copy2(filepath, dest_dir)
                        print(f"  Copied: {filepath.name}")
                    except Exception as e:
                        print(f"  Error copying {filepath}: {e}")
                
                # If we have any PNG files, consider it a success for visualization
                if len(list(dest_dir.glob('*.png'))) > 0:
                    visualization_success = True
                else:
                    visualization_success = False
            else:
                visualization_success = False
                print(f"  No output directory found for {system}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            status = "✓ PASSED" if success else "✗ FAILED"
            vis_status = "✓ VISUALIZED" if visualization_success else "✗ NO VISUALIZATIONS"
            
            print(f"  {status} {vis_status} - {system} ({duration:.2f}s)")
            
            results.append({
                'system': system,
                'test_success': success,
                'visualization_success': visualization_success,
                'duration': duration
            })
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"  Error running test for {system}: {e}")
            
            results.append({
                'system': system,
                'test_success': False,
                'visualization_success': False,
                'duration': duration,
                'error': str(e)
            })
    
    # Create a summary report
    print("\nSummary of DMBD Visualizations:")
    print("-" * 50)
    
    successful_visualizations = sum(1 for r in results if r['visualization_success'])
    
    for result in results:
        status = "✓" if result['visualization_success'] else "✗"
        system = result['system']
        print(f"  {status} {system.capitalize()}")
    
    print(f"\nSuccessfully generated visualizations for {successful_visualizations}/{len(results)} systems")
    
    if successful_visualizations > 0:
        print(f"\nVisualizations are available in: {output_dir}")
        print("Key visualizations to look for:")
        print("  - *_markov_blanket.png: Shows the discovered causal structure")
        print("  - *_assignments.png: Shows variable role assignments (sensor/boundary/internal)")
        print("  - *_sbz_analysis.png: Shows principal component analysis of Markov blanket roles")
        print("  - *_reconstruction.png: Compares original data with DMBD reconstruction")
    
    return 0 if successful_visualizations > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 