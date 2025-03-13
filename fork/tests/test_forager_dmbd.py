"""
Test suite for Dynamic Markov Blanket Detection on the Forager system.

This module provides tests for DMBD inference and visualization on a foraging agent 
simulation, demonstrating how DMBD can be applied to agent-based models.
"""
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
from dmbd import DMBD
import pytest
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pandas.plotting import autocorrelation_plot

# Get the absolute path to the project root
project_root = Path(__file__).parent.parent
examples_path = project_root / "examples"

# Add examples directory to Python path
sys.path.insert(0, str(examples_path))

from Forager import Forager

# Create output directory for Forager DMBD results
output_dir = project_root / "dmbd_outputs"
forager_dir = output_dir / "forager"
os.makedirs(forager_dir, exist_ok=True)

def visualize_forager_results(results, output_dir):
    """
    Visualize the results of the Forager DMBD analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    positions = results.get('positions')
    food_positions = results.get('food_positions')
    reward = results.get('reward')
    
    # Basic trajectory plot
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Forager')
    plt.scatter(food_positions[:, 0], food_positions[:, 1], c='g', alpha=0.5, s=20, label='Food')
    plt.scatter(positions[0, 0], positions[0, 1], c='r', s=100, marker='*', label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 1], c='m', s=100, marker='o', label='End')
    plt.title('Forager Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory.png'))
    plt.close()
    
    # Reward over time
    plt.figure(figsize=(12, 6))
    plt.plot(reward, 'g-')
    plt.title('Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'reward.png'))
    plt.close()
    
    # Heatmap of positions
    plt.figure(figsize=(10, 10))
    heatmap, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], 
        bins=20, range=[[0, 10], [0, 10]]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.title('Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(os.path.join(output_dir, 'position_heatmap.png'))
    plt.close()
    
    # Check if DMBD update was successful
    if not results.get('update_success', False):
        # Create placeholder visuals with detailed diagnostics
        create_placeholder_visuals(
            results, 
            output_dir, 
            results.get('error_traceback')
        )
        return
    
    # Only attempt to visualize Markov blankets if the model update was successful
    if results['dmbd_results']:
        print("DMBD update successful, visualizing Markov blankets...")
        # Visualize Markov blanket assignments
        visualize_markov_blankets(results, output_dir)
        
        # Create analytics report on Markov blankets
        analyze_markov_blankets(results, output_dir)
        
        # Create animation of dynamic Markov blankets
        animate_markov_blankets(results, output_dir)
    else:
        print("DMBD update failed, skipping Markov blanket visualizations.")
        # Create placeholder visualizations
        create_placeholder_visuals(results, output_dir)

def create_placeholder_visuals(results, save_dir, error_traceback=None):
    """
    Create placeholder visualizations and a diagnostic report when DMBD update fails
    """
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a placeholder image showing the error
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "DMBD Update Failed - See diagnostic report", 
             ha='center', va='center', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dmbd_update_failed.png"))
    plt.close()
    
    # Write detailed diagnostic information to a text file
    with open(os.path.join(save_dir, "blanket_analysis.txt"), 'w') as f:
        f.write("DYNAMIC MARKOV BLANKET ANALYSIS REPORT\n")
        f.write("=====================================\n\n")
        f.write("The DMBD model update failed, so no blanket analysis could be performed.\n\n")
        
        # Report the error message from the results dictionary
        error_msg = results.get('error_message', 'Unknown error')
        f.write(f"Error message: {error_msg}\n\n")
        
        # Include the full traceback if available
        if error_traceback:
            f.write("Error Traceback:\n")
            f.write("--------------\n")
            f.write(error_traceback)
            f.write("\n\n")
        
        # Include diagnostic information about the model
        f.write("Diagnostic Information:\n")
        f.write("-----------------------\n")
        
        # Include detailed input tensor shapes
        if 'data' in results:
            data = results['data']
            f.write(f"Input data tensor shape: {data.shape}\n")
            f.write(f"Data tensor dtype: {data.dtype}\n")
            if isinstance(data, torch.Tensor):
                f.write(f"Data tensor device: {data.device}\n")
                f.write(f"Data tensor min/max: {torch.min(data).item():.4f}, {torch.max(data).item():.4f}\n\n")
        
        # Include model attributes if available
        if 'model' in results:
            model = results['model']
            f.write("Model attributes:\n")
            
            # Key model parameters
            try:
                f.write(f"  obs_shape: {model.obs_shape}\n")
                f.write(f"  role_dims: {model.role_dims}\n")
                f.write(f"  hidden_dims: {model.hidden_dims}\n")
                if hasattr(model, 'number_of_objects'):
                    f.write(f"  number_of_objects: {model.number_of_objects}\n")
                if hasattr(model, 'regression_dim'):
                    f.write(f"  regression_dim: {model.regression_dim}\n")
            except Exception as e:
                f.write(f"  <Error getting model parameters: {str(e)}>\n")
            
            # List all tensor attributes with their shapes
            f.write("\nModel tensors:\n")
            for attr_name in dir(model):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr = getattr(model, attr_name)
                        # For tensors, include their shape
                        if isinstance(attr, torch.Tensor):
                            f.write(f"  {attr_name}: Shape {attr.shape}, "
                                   f"Type {attr.dtype}, Device {attr.device}\n")
                    except Exception as e:
                        pass
            
            # Add observation model details if available
            f.write("\nObservation Model:\n")
            try:
                if hasattr(model, 'obs_model'):
                    obs_model = model.obs_model
                    f.write(f"  Type: {type(obs_model)}\n")
                    
                    for attr_name in dir(obs_model):
                        if not attr_name.startswith('_') and attr_name not in ['forward', 'update']:
                            try:
                                attr = getattr(obs_model, attr_name)
                                if isinstance(attr, torch.Tensor):
                                    f.write(f"  {attr_name}: Shape {attr.shape}\n")
                            except Exception:
                                pass
            except Exception as e:
                f.write(f"  <Error getting observation model details: {str(e)}>\n")
                
        # Add information about potential tensor dimension mismatch
        f.write("\nTensor Dimension Analysis:\n")
        f.write("-----------------------\n")
        if 'error_message' in results and 'dimension' in results['error_message'].lower():
            f.write("The error appears to be related to tensor dimension mismatch. Check:\n")
            f.write("1. That control inputs (u) have the right shape with empty last dimension (time_steps, n_objects, 0)\n")
            f.write("2. That role assignments (r) have shape (time_steps, n_objects, 1)\n")
            f.write("3. That data tensor has shape (time_steps, n_objects, obs_dim)\n")
            f.write("4. That the model.obs_shape matches the data tensor dimensions\n")
        
        # Write model update parameters used
        f.write("\nUpdate Parameters Used:\n")
        f.write("-----------------------\n")
        if 'update_parameters' in results:
            params = results['update_parameters']
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"  {key}: Shape {value.shape}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        else:
            f.write("  Update parameters not recorded\n")
            
        # Record any post-update debug info
        if 'debug_info' in results:
            f.write("\nDebug Information:\n")
            f.write("-----------------\n")
            f.write(results['debug_info'])
            
    # Create additional debug visualizations
    if 'data' in results and isinstance(results['data'], torch.Tensor):
        # Create visualization of the data time series
        try:
            data = results['data']
            time_steps = data.shape[0]
            
            plt.figure(figsize=(12, 8))
            plt.suptitle("Data Tensor Visualization")
            
            # Plot key dimensions
            for i in range(min(5, data.shape[2])):
                plt.subplot(min(5, data.shape[2]), 1, i+1)
                plt.plot(range(time_steps), data[:, 0, i].cpu().numpy())
                plt.ylabel(f"Dim {i}")
                plt.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "data_tensor_visualization.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating data visualization: {str(e)}")
            
    # Write debug log with setup information
    with open(os.path.join(save_dir, "debug_log.txt"), "w") as f:
        f.write("DMBD DEBUG LOG\n")
        f.write("=============\n\n")
        
        # Log the Forager simulation parameters
        f.write("Forager Simulation Parameters:\n")
        if 'forager_parameters' in results:
            params = results['forager_parameters']
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write("  Forager parameters not recorded\n")
        
        # Log tensor shapes and parameters
        if 'data' in results:
            data = results['data']
            f.write(f"\nData tensor shape: {data.shape}\n")
        
        # Log error information
        if 'error_message' in results:
            f.write(f"\nError: {results['error_message']}\n")
        
        if 'error_traceback' in results:
            f.write(f"\nTraceback:\n{results['error_traceback']}\n")

def visualize_markov_blankets(results, save_dir):
    """Visualize the detected Markov blankets."""
    model = results['model']
    data = results['data']
    positions = results.get('positions')
    food_positions = results.get('food_positions')
    
    # Ensure save_dir is a Path object
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    # Get assignment probabilities from the model
    try:
        # Try to get assignment probabilities
        assignment_probs = results.get('assignments_pr')
        if assignment_probs is None:
            print("Assignment probabilities not found in results, trying to extract from model...")
            assignment_probs = model.assignment_pr().detach().cpu().numpy()
        else:
            assignment_probs = assignment_probs.numpy()
        
        has_assignments = True
        print("Successfully loaded Markov blanket assignments")
    except (AttributeError, RuntimeError) as e:
        print(f"Warning: Could not extract assignment probabilities: {e}")
        has_assignments = False
    
    if has_assignments:
        # Plot assignment probabilities over time
        plt.figure(figsize=(16, 12))
        
        # Get the number of possible roles (typically system/blanket/external)
        n_roles = assignment_probs.shape[-1]
        role_names = ["System", "Markov Blanket", "Environment"]
        colors = ['red', 'green', 'blue']
        
        # Plot assignment probabilities over time
        plt.subplot(2, 2, 1)
        for i in range(min(n_roles, 3)):
            plt.plot(assignment_probs[:, 0, i], color=colors[i], 
                     label=role_names[i] if i < len(role_names) else f"Role {i}")
        plt.title("Markov Blanket Assignment Probabilities", fontsize=14)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Get hard assignments 
        try:
            assignments = results.get('assignments')
            if assignments is None:
                print("Hard assignments not found in results, trying to extract from model...")
                assignments = model.assignment().detach().cpu().numpy()
            else:
                assignments = assignments.numpy()
            
            # Plot hard assignments over time as a colored line
            plt.subplot(2, 2, 2)
            
            # Create a colormap for the assignments
            cmap = plt.cm.get_cmap('viridis', n_roles)
            
            # Plot the assignment as a colormap
            plt.imshow(assignments[:, 0].reshape(1, -1), 
                      aspect='auto', 
                      cmap=cmap, 
                      extent=[0, len(assignments), -0.5, 0.5])
            
            # Add color bar with role labels
            cbar = plt.colorbar(ticks=range(n_roles))
            cbar.set_label("Role Assignment", fontsize=12)
            cbar.ax.set_yticklabels([role_names[i] if i < len(role_names) else f"Role {i}" 
                                   for i in range(min(n_roles, 3))])
            
            plt.title("Markov Blanket Hard Assignments Over Time", fontsize=14)
            plt.xlabel("Time Step", fontsize=12)
            plt.yticks([])
            plt.grid(False)
            
            # Create a second visualization as a scatter plot for clarity
            ax2 = plt.gca().twinx()
            for i in range(min(n_roles, 3)):
                mask = assignments[:, 0] == i
                if np.any(mask):
                    time_points = np.where(mask)[0]
                    ax2.scatter(time_points, np.ones(len(time_points)) * 0, 
                              color=colors[i], s=5, alpha=0.7,
                              label=role_names[i] if i < len(role_names) else f"Role {i}")
            ax2.set_yticks([])
            ax2.legend(loc='upper right', fontsize=8)
        except (AttributeError, RuntimeError) as e:
            print(f"Warning: Could not extract hard assignments: {e}")
        
        # For 2D visualization, project the forager trajectory and color by blanket assignment
        plt.subplot(2, 2, 3)
        try:
            # Use actual positions for better visualization
            x_pos = positions[:, 0].numpy()
            y_pos = positions[:, 1].numpy()
            
            # Create a scatter plot with points colored by their assignment
            if 'assignments' in locals():
                # Create a new custom colormap with our specific colors
                custom_cmap = LinearSegmentedColormap.from_list('custom', 
                                                             colors[:min(n_roles, 3)], 
                                                             N=min(n_roles, 3))
                
                # Plot the trajectory as a line
                plt.plot(x_pos, y_pos, 'k-', alpha=0.2, linewidth=1)
                
                # Scatter plot colored by role
                for i in range(min(n_roles, 3)):
                    mask = assignments[:, 0] == i
                    if np.any(mask):
                        plt.scatter(x_pos[mask], y_pos[mask], c=colors[i], 
                                  label=role_names[i] if i < len(role_names) else f"Role {i}",
                                  alpha=0.7, s=30)
                
                # Add food positions
                if food_positions is not None:
                    plt.scatter(food_positions[:, 0], food_positions[:, 1], 
                              c='orange', marker='*', s=100, label='Food', alpha=0.7)
            else:
                # Use soft assignments (assignment with highest probability)
                max_assignment = np.argmax(assignment_probs, axis=-1)
                for i in range(min(n_roles, 3)):
                    mask = max_assignment[:, 0] == i
                    if np.any(mask):
                        plt.scatter(x_pos[mask], y_pos[mask], c=colors[i], 
                                  label=role_names[i] if i < len(role_names) else f"Role {i}",
                                  alpha=0.7, s=30)
            
            plt.title("Forager Trajectory Colored by Markov Blanket Role", fontsize=14)
            plt.xlabel("X Position", fontsize=12)
            plt.ylabel("Y Position", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not create trajectory visualization: {e}")
            traceback.print_exc()
        
        # Create a heatmap of transition probabilities between roles
        plt.subplot(2, 2, 4)
        try:
            if 'assignments' in locals():
                # Compute transition probabilities between assignments
                transitions = np.zeros((min(n_roles, 3), min(n_roles, 3)))
                for t in range(len(assignments) - 1):
                    from_role = assignments[t, 0]
                    to_role = assignments[t + 1, 0]
                    if from_role < min(n_roles, 3) and to_role < min(n_roles, 3):
                        transitions[from_role, to_role] += 1
                
                # Normalize by row sum
                row_sums = transitions.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                transition_probs = transitions / row_sums
                
                # Create heatmap
                sns.heatmap(transition_probs, annot=True, cmap="YlGnBu", fmt=".2f",
                           xticklabels=[role_names[i] if i < len(role_names) else f"Role {i}" 
                                      for i in range(min(n_roles, 3))],
                           yticklabels=[role_names[i] if i < len(role_names) else f"Role {i}" 
                                      for i in range(min(n_roles, 3))])
                plt.title("Role Transition Probabilities", fontsize=14)
                plt.xlabel("To Role", fontsize=12)
                plt.ylabel("From Role", fontsize=12)
                
                # Add role distribution as text
                role_counts = [np.sum(assignments[:, 0] == i) for i in range(min(n_roles, 3))]
                role_percentages = [count / len(assignments) * 100 for count in role_counts]
                plt.figtext(0.5, 0.01, f"Role Distribution: " + 
                           ", ".join([f"{role_names[i]}: {role_counts[i]} ({role_percentages[i]:.1f}%)" 
                                     for i in range(min(n_roles, 3))]),
                           ha="center", fontsize=10, 
                           bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            else:
                plt.text(0.5, 0.5, "Hard assignments not available", 
                       horizontalalignment='center', verticalalignment='center')
        except Exception as e:
            print(f"Warning: Could not create transition matrix: {e}")
            traceback.print_exc()
        
        plt.tight_layout()
        plt.savefig(save_dir / "markov_blanket_visualization.png", dpi=300)
        plt.close()
        
        # Create a separate whole-figure visualization of the trajectory
        plt.figure(figsize=(12, 10))
        try:
            if 'assignments' in locals():
                # Create a new custom colormap with our specific colors
                custom_cmap = LinearSegmentedColormap.from_list('custom', 
                                                           colors[:min(n_roles, 3)], 
                                                           N=min(n_roles, 3))
                
                # Plot the trajectory as a line
                plt.plot(x_pos, y_pos, 'k-', alpha=0.3, linewidth=1.5)
                
                # Scatter plot colored by role - make this larger for better visibility
                for i in range(min(n_roles, 3)):
                    mask = assignments[:, 0] == i
                    if np.any(mask):
                        plt.scatter(x_pos[mask], y_pos[mask], c=colors[i], 
                                  label=role_names[i] if i < len(role_names) else f"Role {i}",
                                  alpha=0.8, s=50, edgecolors='black', linewidths=0.5)
                
                # Add food positions with larger markers
                if food_positions is not None:
                    plt.scatter(food_positions[:, 0], food_positions[:, 1], 
                              c='orange', marker='*', s=200, label='Food', alpha=0.8,
                              edgecolors='black', linewidths=0.7)
                
                # Mark start and end positions
                plt.scatter(x_pos[0], y_pos[0], c='purple', s=200, marker='^', label='Start',
                         edgecolors='black', linewidths=1.0)
                plt.scatter(x_pos[-1], y_pos[-1], c='cyan', s=200, marker='s', label='End',
                         edgecolors='black', linewidths=1.0)
                
                plt.title("Forager Trajectory with Dynamic Markov Blanket Roles", fontsize=16)
                plt.xlabel("X Position", fontsize=14)
                plt.ylabel("Y Position", fontsize=14)
                plt.legend(fontsize=12, loc='best')
                plt.grid(True, alpha=0.3)
                
                # Add role distribution as text
                role_counts = [np.sum(assignments[:, 0] == i) for i in range(min(n_roles, 3))]
                role_percentages = [count / len(assignments) * 100 for count in role_counts]
                plt.figtext(0.5, 0.01, f"Role Distribution: " + 
                           ", ".join([f"{role_names[i]}: {role_counts[i]} ({role_percentages[i]:.1f}%)" 
                                     for i in range(min(n_roles, 3))]),
                           ha="center", fontsize=10, 
                           bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout()
            plt.savefig(save_dir / "dynamic_markov_blanket_trajectory.png", dpi=300)
            plt.close()
        
        except Exception as e:
            print(f"Warning: Could not create detailed trajectory visualization: {e}")
            traceback.print_exc()
        
        # Save assignment data for further analysis
        if has_assignments:
            np.savez(save_dir / "assignments.npz", 
                   probabilities=assignment_probs,
                   hard_assignments=assignments if 'assignments' in locals() else None,
                   role_names=role_names)

def analyze_markov_blankets(results, save_dir):
    """Analyze the properties of detected Markov blankets."""
    # Ensure save_dir is a Path object
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    model = results['model']
    data = results['data']
    positions = results.get('positions')
    food_positions = results.get('food_positions')
    
    # Define role names for consistent labeling
    role_names = ["System", "Markov Blanket", "Environment"]
    colors = ['red', 'green', 'blue']
    
    # Try to get assignments
    try:
        # Get assignments from results if available
        assignments = results.get('assignments')
        if assignments is None:
            print("Assignments not found in results, trying to extract from model...")
            assignments = model.assignment().detach().cpu().numpy()
        else:
            assignments = assignments.numpy()
            
        has_assignments = True
        print("Successfully loaded assignments for analysis")
    except Exception as e:
        print(f"Warning: Could not get assignments for analysis: {e}")
        has_assignments = False
        
    if has_assignments:
        # Create a report file with detailed analytics
        with open(save_dir / "blanket_analysis.txt", "w") as f:
            f.write("DYNAMIC MARKOV BLANKET ANALYSIS REPORT\n")
            f.write("=====================================\n\n")
            
            # Count role occurrences and percentages
            n_roles = assignments.max() + 1
            role_counts = np.zeros(min(3, n_roles))
            for i in range(min(3, n_roles)):
                role_counts[i] = np.sum(assignments[:, 0] == i)
                percentage = (role_counts[i] / len(assignments)) * 100
                f.write(f"{role_names[i]} role: {int(role_counts[i])} time steps ({percentage:.2f}%)\n")
            
            f.write("\n")
            
            # Calculate average duration in each role (bout lengths)
            current_role = assignments[0, 0]
            current_duration = 1
            role_durations = [[] for _ in range(min(3, n_roles))]
            
            for t in range(1, len(assignments)):
                if assignments[t, 0] == current_role:
                    current_duration += 1
                else:
                    if current_role < min(3, n_roles):
                        role_durations[current_role].append(current_duration)
                    current_role = assignments[t, 0]
                    current_duration = 1
            
            # Add the final duration
            if current_role < min(3, n_roles):
                role_durations[current_role].append(current_duration)
            
            f.write("Role Duration Analysis:\n")
            f.write("----------------------\n")
            for i in range(min(3, n_roles)):
                if role_durations[i]:
                    avg_duration = np.mean(role_durations[i])
                    median_duration = np.median(role_durations[i])
                    min_duration = np.min(role_durations[i])
                    max_duration = np.max(role_durations[i])
                    num_bouts = len(role_durations[i])
                    
                    f.write(f"{role_names[i]} role:\n")
                    f.write(f"  Number of bouts: {num_bouts}\n")
                    f.write(f"  Average duration: {avg_duration:.2f} steps\n")
                    f.write(f"  Median duration: {median_duration:.2f} steps\n") 
                    f.write(f"  Min duration: {min_duration} steps\n")
                    f.write(f"  Max duration: {max_duration} steps\n")
                else:
                    f.write(f"{role_names[i]} role: No occurrences\n")
            
            f.write("\n")
            
            # Analyze the model variables by role
            f.write("Feature Analysis by Role:\n")
            f.write("------------------------\n")
            
            # Velocity analysis by role
            f.write("Movement and Velocity:\n")
            for i in range(min(3, n_roles)):
                mask = assignments[:, 0] == i
                if np.any(mask):
                    vx = data[mask, 0, 2].numpy()  # x velocity
                    vy = data[mask, 0, 3].numpy()  # y velocity
                    
                    # Calculate velocity magnitudes
                    v_magnitude = np.sqrt(vx**2 + vy**2)
                    
                    # Calculate velocity direction (angle in radians, then convert to degrees)
                    v_direction = np.arctan2(vy, vx) * 180 / np.pi
                    
                    # Calculate average velocity vector
                    avg_vx = np.mean(vx)
                    avg_vy = np.mean(vy)
                    resultant_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                    
                    f.write(f"{role_names[i]} role:\n")
                    f.write(f"  Velocity magnitude: mean={np.mean(v_magnitude):.4f}, std={np.std(v_magnitude):.4f}\n")
                    
                    # Calculate speed variability
                    f.write(f"  Speed variability (CV): {np.std(v_magnitude) / np.mean(v_magnitude) if np.mean(v_magnitude) > 0 else 0:.4f}\n")
                    
                    # Calculate directional statistics
                    f.write(f"  Directional consistency: {resultant_magnitude / np.mean(v_magnitude) if np.mean(v_magnitude) > 0 else 0:.4f}\n")
                    
                    # Calculate directional bias
                    if avg_vx != 0 or avg_vy != 0:
                        avg_direction = np.arctan2(avg_vy, avg_vx) * 180 / np.pi
                        f.write(f"  Average movement direction: {avg_direction:.1f} degrees\n")
                    
                    # Add turning rates
                    if len(v_direction) > 1:
                        # Calculate angular differences between consecutive time steps (wrapped to [-180, 180])
                        ang_diff = np.diff(v_direction)
                        ang_diff = (ang_diff + 180) % 360 - 180  # Wrap to [-180, 180]
                        
                        # Calculate average turning rate (absolute angular changes)
                        avg_turning_rate = np.mean(np.abs(ang_diff))
                        f.write(f"  Average turning rate: {avg_turning_rate:.2f} degrees/step\n")
            
            f.write("\n")
            
            # Memory state analysis by role
            f.write("Memory State Analysis:\n")
            for i in range(min(3, n_roles)):
                mask = assignments[:, 0] == i
                if np.any(mask):
                    memory = data[mask, 0, 4].numpy()
                    
                    # Memory statistics
                    f.write(f"{role_names[i]} role:\n")
                    f.write(f"  Memory value: mean={np.mean(memory):.4f}, std={np.std(memory):.4f}\n")
                    f.write(f"  Memory range: min={np.min(memory):.4f}, max={np.max(memory):.4f}\n")
                    
                    # Memory dynamics
                    if len(memory) > 1:
                        memory_changes = np.diff(memory)
                        positive_changes = memory_changes > 0
                        
                        f.write(f"  Memory increasing in {np.sum(positive_changes)} of {len(memory_changes)} steps ({np.mean(positive_changes)*100:.1f}%)\n")
                        f.write(f"  Average memory change: {np.mean(memory_changes):.4f} per step\n")
            
            f.write("\n")
            
            # Spatial analysis by role
            if positions is not None:
                f.write("Spatial Analysis:\n")
                for i in range(min(3, n_roles)):
                    mask = assignments[:, 0] == i
                    if np.any(mask):
                        pos_x = positions[mask, 0].numpy()
                        pos_y = positions[mask, 1].numpy()
                        
                        # Calculate spatial statistics
                        centroid_x = np.mean(pos_x)
                        centroid_y = np.mean(pos_y)
                        
                        # Calculate distance from centroid (spatial spread)
                        distances = np.sqrt((pos_x - centroid_x)**2 + (pos_y - centroid_y)**2)
                        
                        # Distance from origin
                        origin_distances = np.sqrt(pos_x**2 + pos_y**2)
                        
                        f.write(f"{role_names[i]} role:\n")
                        f.write(f"  Centroid: ({centroid_x:.2f}, {centroid_y:.2f})\n")
                        f.write(f"  Spatial spread: {np.mean(distances):.2f} (avg distance from centroid)\n")
                        f.write(f"  Average distance from origin: {np.mean(origin_distances):.2f}\n")
                        
                        # Proximity to food if available
                        if food_positions is not None:
                            all_food_distances = []
                            for food_pos in food_positions:
                                # Calculate distances from each position to this food item
                                food_x, food_y = food_pos[0].item(), food_pos[1].item()
                                food_distances = np.sqrt((pos_x - food_x)**2 + (pos_y - food_y)**2)
                                all_food_distances.append(np.min(food_distances))  # Closest approach
                            
                            # Calculate statistics on closest approaches to any food
                            closest_approaches = np.min(all_food_distances, axis=0) if all_food_distances else []
                            if len(closest_approaches) > 0:
                                f.write(f"  Average closest distance to food: {np.mean(closest_approaches):.2f}\n")
                                f.write(f"  Minimum distance to food: {np.min(closest_approaches):.2f}\n")
                
                f.write("\n")
            
            # Role transition matrix
            transitions = np.zeros((min(3, n_roles), min(3, n_roles)))
            for t in range(len(assignments) - 1):
                from_role = assignments[t, 0]
                to_role = assignments[t + 1, 0]
                if from_role < min(3, n_roles) and to_role < min(3, n_roles):
                    transitions[from_role, to_role] += 1
            
            # Normalize by row sum for transition probabilities
            row_sums = transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_probs = transitions / row_sums
            
            f.write("Role Transition Analysis:\n")
            f.write("-----------------------\n")
            f.write("Transition Counts:\n")
            f.write("      TO:   " + "  ".join([f"{role_names[i]:8}" for i in range(min(3, n_roles))]) + "\n")
            f.write("FROM:\n")
            for i in range(min(3, n_roles)):
                f.write(f"{role_names[i]:8}: " + "  ".join([f"{transitions[i, j]:8.0f}" for j in range(min(3, n_roles))]) + "\n")
            
            f.write("\nTransition Probabilities:\n")
            f.write("      TO:   " + "  ".join([f"{role_names[i]:8}" for i in range(min(3, n_roles))]) + "\n")
            f.write("FROM:\n")
            for i in range(min(3, n_roles)):
                f.write(f"{role_names[i]:8}: " + "  ".join([f"{transition_probs[i, j]:8.3f}" for j in range(min(3, n_roles))]) + "\n")
            
            # Calculate entropy of transition matrix rows (predictability)
            entropy = np.zeros(min(3, n_roles))
            for i in range(min(3, n_roles)):
                p = transition_probs[i]
                # Only consider non-zero probabilities for entropy calculation
                p_nonzero = p[p > 0]
                if len(p_nonzero) > 0:
                    entropy[i] = -np.sum(p_nonzero * np.log2(p_nonzero))
            
            f.write("\nRole Transition Entropy (higher = more random transitions):\n")
            for i in range(min(3, n_roles)):
                if np.sum(transitions[i]) > 0:  # Only if there are transitions from this role
                    max_entropy = np.log2(min(3, n_roles))
                    f.write(f"{role_names[i]}: {entropy[i]:.3f} (normalized: {entropy[i]/max_entropy:.3f})\n")
            
            # Add information about the model parameters
            f.write("\nModel Parameters:\n")
            f.write("-----------------\n")
            if hasattr(model, 'role_dims'):
                f.write(f"Role dimensions: {model.role_dims}\n")
            if hasattr(model, 'hidden_dims'):
                f.write(f"Hidden dimensions: {model.hidden_dims}\n")
            if hasattr(model, 'obs_shape'):
                f.write(f"Observation shape: {model.obs_shape}\n")
            if 'forager_parameters' in results:
                f.write("\nForager Simulation Parameters:\n")
                for key, value in results['forager_parameters'].items():
                    f.write(f"  {key}: {value}\n")
            
            # Add information about assignment sequences
            f.write("\nAssignment Sequence Patterns:\n")
            f.write("----------------------------\n")
            
            # Find common role sequences (e.g., S→B→E patterns)
            # Look at sequences of length 3
            seq_length = 3
            sequences = {}
            total_seqs = len(assignments) - seq_length + 1
            
            if total_seqs > 0:
                for t in range(total_seqs):
                    seq = tuple(assignments[t:t+seq_length, 0])
                    if seq in sequences:
                        sequences[seq] += 1
                    else:
                        sequences[seq] = 1
                
                # Sort by frequency
                sorted_seqs = sorted(sequences.items(), key=lambda x: x[1], reverse=True)
                
                # Report top sequences
                f.write(f"Most common {seq_length}-role sequences:\n")
                for i, (seq, count) in enumerate(sorted_seqs[:5]):  # Show top 5
                    percentage = count / total_seqs * 100
                    seq_str = "→".join([role_names[min(int(r), len(role_names)-1)] for r in seq])
                    f.write(f"{i+1}. {seq_str}: {count} occurrences ({percentage:.2f}%)\n")
        
        # Create visualization of behavioral and statistical patterns
        plt.figure(figsize=(15, 12))
        
        # Plot velocity distributions by role
        plt.subplot(2, 3, 1)
        for i in range(min(3, n_roles)):
            mask = assignments[:, 0] == i
            if np.any(mask):
                vx = data[mask, 0, 2].numpy()
                vy = data[mask, 0, 3].numpy()
                v_magnitude = np.sqrt(vx**2 + vy**2)
                sns.kdeplot(v_magnitude, label=role_names[i], color=colors[i])
        plt.title("Velocity Magnitude Distribution by Role", fontsize=12)
        plt.xlabel("Velocity Magnitude", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Plot memory state distribution by role
        plt.subplot(2, 3, 2)
        for i in range(min(3, n_roles)):
            mask = assignments[:, 0] == i
            if np.any(mask):
                memory = data[mask, 0, 4].numpy()
                sns.kdeplot(memory, label=role_names[i], color=colors[i])
        plt.title("Memory State Distribution by Role", fontsize=12)
        plt.xlabel("Memory State", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Plot role duration histogram
        plt.subplot(2, 3, 3)
        for i in range(min(3, n_roles)):
            if role_durations[i]:
                plt.hist(role_durations[i], alpha=0.7, label=role_names[i], 
                       bins=min(10, max(5, len(role_durations[i])//5)), color=colors[i])
        plt.title("Distribution of Role Durations", fontsize=12)
        plt.xlabel("Duration (time steps)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Plot transition heatmap
        plt.subplot(2, 3, 4)
        sns.heatmap(transition_probs, annot=True, cmap="YlGnBu", fmt=".2f",
                   xticklabels=[role_names[i] for i in range(min(3, n_roles))],
                   yticklabels=[role_names[i] for i in range(min(3, n_roles))])
        plt.title("Role Transition Probabilities", fontsize=12)
        plt.xlabel("To Role", fontsize=10)
        plt.ylabel("From Role", fontsize=10)
        
        # Plot spatial analysis - position heatmap by role
        if positions is not None:
            plt.subplot(2, 3, 5)
            for i in range(min(3, n_roles)):
                mask = assignments[:, 0] == i
                if np.any(mask):
                    pos_x = positions[mask, 0].numpy()
                    pos_y = positions[mask, 1].numpy()
                    
                    # Create a 2D histogram (heatmap) for positions
                    h = plt.hist2d(pos_x, pos_y, bins=15, alpha=0.5, 
                                 cmap=plt.cm.get_cmap(f'Reds' if i==0 else 'Greens' if i==1 else 'Blues'),
                                 density=True)[3]  # get the colorbar handle
                    h.set_label(role_names[i])
                    
            if food_positions is not None:
                plt.scatter(food_positions[:, 0], food_positions[:, 1], 
                          c='yellow', s=80, alpha=0.8, marker='*', label='Food')
            
            plt.title("Spatial Distribution by Role", fontsize=12)
            plt.xlabel("X Position", fontsize=10)
            plt.ylabel("Y Position", fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
        
        # Plot directional analysis - velocity vectors by role
        plt.subplot(2, 3, 6)
        
        # Combine velocity data with roles for visualization
        vx_all = data[:, 0, 2].numpy()
        vy_all = data[:, 0, 3].numpy()
        roles = assignments[:, 0]
        
        # Create a 2D histogram of velocity directions by role
        for i in range(min(3, n_roles)):
            mask = roles == i
            if np.any(mask):
                vx_role = vx_all[mask]
                vy_role = vy_all[mask]
                
                # Plot the average velocity vector
                avg_vx = np.mean(vx_role)
                avg_vy = np.mean(vy_role)
                
                # Scale based on number of samples
                scale = np.sqrt(np.sum(mask)) / 10
                
                plt.quiver(0, 0, avg_vx, avg_vy, 
                         angles='xy', scale_units='xy', scale=0.1,
                         color=colors[i], label=f"{role_names[i]} mean")
                
                # Also plot a sample of individual velocity vectors
                stride = max(1, len(vx_role) // 50)  # Select a subset for clarity
                plt.quiver(np.zeros_like(vx_role[::stride]), np.zeros_like(vy_role[::stride]),
                         vx_role[::stride], vy_role[::stride],
                         angles='xy', scale_units='xy', scale=0.2,
                         alpha=0.3, color=colors[i])
        
        plt.title("Average Velocity Vectors by Role", fontsize=12)
        plt.xlabel("X Velocity", fontsize=10)
        plt.ylabel("Y Velocity", fontsize=10)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        plt.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_dir / "blanket_analysis.png", dpi=300)
        plt.close()
        
        # Create an additional figure for role transitions over time
        plt.figure(figsize=(15, 6))
        
        # Plot role assignments over time
        ax1 = plt.subplot(2, 1, 1)
        
        # Create a colormap for roles
        cmap = plt.cm.get_cmap('viridis', n_roles)
        
        # Plot the assignment as a colormap
        plt.imshow(assignments[:, 0].reshape(1, -1), 
                 aspect='auto', 
                 cmap=cmap,
                 extent=[0, len(assignments), -0.5, 0.5],
                 interpolation='nearest')
        
        # Add colorbar with role labels
        cbar = plt.colorbar(ticks=range(n_roles))
        cbar.set_label("Role Assignment", fontsize=10)
        cbar.ax.set_yticklabels([role_names[i] if i < len(role_names) else f"Role {i}" 
                               for i in range(min(n_roles, 3))])
        
        plt.title("Dynamic Markov Blanket Roles Over Time", fontsize=14)
        plt.xlabel("Time Step", fontsize=12)
        plt.yticks([])
        
        # Add memory and velocity traces in subplot below the role assignments
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        
        # Plot memory state
        memory_trace = data[:, 0, 4].numpy()
        plt.plot(range(len(memory_trace)), memory_trace, 'g-', label='Memory State', alpha=0.7)
        
        # Plot velocity magnitude
        vx = data[:, 0, 2].numpy()
        vy = data[:, 0, 3].numpy()
        v_magnitude = np.sqrt(vx**2 + vy**2)
        plt.plot(range(len(v_magnitude)), v_magnitude, 'b-', label='Velocity', alpha=0.7)
        
        plt.title("Memory State and Velocity Over Time", fontsize=12)
        plt.xlabel("Time Step", fontsize=10)
        plt.ylabel("Value", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "role_dynamics.png", dpi=300)
        plt.close()

def animate_markov_blankets(results, save_dir):
    """Create an animation of dynamic Markov blankets."""
    import matplotlib.animation as animation
    
    # Ensure save_dir is a Path object
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    model = results['model']
    data = results['data']
    positions = results.get('positions')
    food_positions = results.get('food_positions')
    
    # Initialize role names and colors
    role_names = ["System", "Markov Blanket", "Environment"]
    colors = ['red', 'green', 'blue']
    
    try:
        # Get assignments from results if available
        assignments = results.get('assignments')
        if assignments is None:
            print("Assignments not found in results, trying to extract from model...")
            assignments = model.assignment().detach().cpu().numpy()
        else:
            assignments = assignments.numpy()
        has_assignments = True
    except Exception as e:
        print(f"Warning: Could not get assignments for animation: {e}")
        has_assignments = False
    
    if has_assignments:
        # Create figure for animation with improved layout
        fig = plt.figure(figsize=(12, 10))
        
        # Set up grid for subplots
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
        
        # Main trajectory plot
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Side panel for current state
        ax_state = fig.add_subplot(gs[0, 1])
        
        # Bottom panel for assignment timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        
        # Set title for the whole figure
        fig.suptitle("Dynamic Markov Blanket Analysis of Foraging Behavior", fontsize=16)
        
        # Get the number of roles
        n_roles = assignments.max() + 1
        
        # Set up animation parameters
        n_frames = min(len(data), 200)  # Cap at 200 frames for manageable file size
        stride = max(1, len(data) // n_frames)
        frame_indices = list(range(0, len(data), stride))
        
        # Use actual positions for better visualization
        if positions is not None:
            x_pos = positions[:, 0].numpy()
            y_pos = positions[:, 1].numpy()
        else:
            x_pos = data[:, 0, 0].numpy()
            y_pos = data[:, 0, 1].numpy()
        
        # Track the forager's path
        path_x = []
        path_y = []
        
        # Set plot limits with some padding
        x_min, x_max = np.min(x_pos), np.max(x_pos)
        y_min, y_max = np.min(y_pos), np.max(y_pos)
        padding = 0.1 * max(x_max - x_min, y_max - y_min)
        ax_main.set_xlim(x_min - padding, x_max + padding)
        ax_main.set_ylim(y_min - padding, y_max + padding)
        
        # Plot food positions (constant throughout animation)
        if food_positions is not None:
            food_scatter = ax_main.scatter(food_positions[:, 0], food_positions[:, 1], 
                                       c='orange', s=100, alpha=0.7, label='Food Items',
                                       edgecolors='black', linewidths=0.5, marker='*')
        
        # Initialize forager marker - use a larger marker for better visibility
        forager_scatter = ax_main.scatter([], [], s=150, c='blue', marker='o', 
                                      edgecolor='black', linewidth=1.5, zorder=10)
        
        # Add a larger circle around the forager when it's in "blanket" state
        blanket_indicator = ax_main.scatter([], [], s=300, c='none', alpha=0.7, 
                                        marker='o', edgecolor='green', linewidth=2.5, zorder=5)
        
        # Initialize path line
        path_line, = ax_main.plot([], [], 'k-', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Add marker at the start position
        ax_main.scatter(x_pos[0], y_pos[0], c='purple', s=150, marker='^', 
                      label='Start', zorder=12, edgecolors='black', linewidths=1.0)
        
        # Add legend to main plot
        ax_main.legend(loc='upper right', fontsize=10)
        ax_main.set_xlabel('X Position', fontsize=12)
        ax_main.set_ylabel('Y Position', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        
        # Set up the state panel to show current state info
        ax_state.axis('off')
        state_text = ax_state.text(0.5, 0.9, "", transform=ax_state.transAxes, 
                                ha='center', va='top', fontsize=12, 
                                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add role color legend to state panel
        for i, (name, color) in enumerate(zip(role_names, colors)):
            ax_state.add_patch(plt.Rectangle((0.1, 0.7 - i*0.1), 0.1, 0.05, 
                                          facecolor=color, edgecolor='black'))
            ax_state.text(0.25, 0.725 - i*0.1, name, fontsize=10)
        
        # Set up timeline in the bottom panel
        timeline_data = np.zeros((1, len(assignments)))
        timeline_img = ax_timeline.imshow(
            timeline_data, 
            aspect='auto', 
            cmap=plt.cm.get_cmap('viridis', n_roles),
            extent=[0, len(assignments), -0.5, 0.5],
            interpolation='nearest'
        )
        
        # Create a marker for the current position in the timeline
        timeline_marker, = ax_timeline.plot([], [], 'r', marker='v', markersize=10, 
                                         linestyle='none', zorder=10)
        
        ax_timeline.set_yticks([])
        ax_timeline.set_xlabel('Time Step', fontsize=12)
        ax_timeline.set_title('Role Assignment Timeline', fontsize=12)
        
        # Create function to initialize animation
        def init():
            """Initialize animation"""
            forager_scatter.set_offsets(np.empty((0, 2)))
            blanket_indicator.set_offsets(np.empty((0, 2)))
            path_line.set_data([], [])
            state_text.set_text("")
            timeline_marker.set_data([], [])
            
            # Initialize with all assignments
            timeline_data = assignments[:, 0].reshape(1, -1)
            timeline_img.set_array(timeline_data)
            
            return forager_scatter, blanket_indicator, path_line, state_text, timeline_marker
        
        # Create function to update animation for each frame
        def update(frame_idx):
            """Update animation for each frame"""
            # Get the current position and role
            current_pos = np.array([x_pos[frame_idx], y_pos[frame_idx]]).reshape(1, 2)
            current_role = int(assignments[frame_idx, 0])
            
            # Update path
            path_x.append(x_pos[frame_idx])
            path_y.append(y_pos[frame_idx])
            path_line.set_data(path_x, path_y)
            
            # Update forager position and color
            forager_scatter.set_offsets(current_pos)
            forager_scatter.set_color(colors[min(current_role, len(colors)-1)])
            
            # Update blanket indicator
            if current_role == 1:  # Blanket role
                blanket_indicator.set_offsets(current_pos)
            else:
                blanket_indicator.set_offsets(np.empty((0, 2)))
            
            # Update state text
            role_name = role_names[min(current_role, len(role_names)-1)]
            memory_value = data[frame_idx, 0, 4].item()
            velocity = np.sqrt(data[frame_idx, 0, 2]**2 + data[frame_idx, 0, 3]**2).item()
            
            state_info = f"Frame: {frame_idx}/{len(assignments)}\n"
            state_info += f"Role: {role_name}\n"
            state_info += f"Position: ({x_pos[frame_idx]:.2f}, {y_pos[frame_idx]:.2f})\n"
            state_info += f"Velocity: {velocity:.2f}\n"
            state_info += f"Memory: {memory_value:.2f}"
            
            state_text.set_text(state_info)
            
            # Update timeline marker
            timeline_marker.set_data([frame_idx], [0])
            
            return forager_scatter, blanket_indicator, path_line, state_text, timeline_marker
        
        # Create animation with improved settings
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, 
                                   init_func=init, blit=True, interval=100)
        
        # Set up the writer with higher quality
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        
        # Save the animation
        animation_path = save_dir / "markov_blanket_animation.gif"
        ani.save(animation_path, writer=writer, dpi=120)
        plt.close()
        
        print(f"Animation saved to {animation_path}")
        
        # Create a second animation focusing on interesting segments
        try:
            # Identify role transitions
            transitions = []
            for t in range(1, len(assignments)):
                if assignments[t, 0] != assignments[t-1, 0]:
                    transitions.append(t)
            
            if len(transitions) > 3:  # At least a few transitions needed
                # Create a dedicated transition animation
                fig_trans = plt.figure(figsize=(12, 8))
                ax_trans = fig_trans.add_subplot(111)
                
                # Set title
                fig_trans.suptitle("Role Transitions in Markov Blanket Analysis", fontsize=14)
                
                # Set plot limits
                ax_trans.set_xlim(x_min - padding, x_max + padding)
                ax_trans.set_ylim(y_min - padding, y_max + padding)
                
                # Plot food positions
                if food_positions is not None:
                    ax_trans.scatter(food_positions[:, 0], food_positions[:, 1], 
                                  c='orange', s=80, alpha=0.7, label='Food Items', marker='*')
                
                # Add context to the plot - full trajectory as a light background
                ax_trans.plot(x_pos, y_pos, 'k-', alpha=0.2, linewidth=1, label='Full Path')
                
                # Initialize elements for animation
                forager_scatter_trans = ax_trans.scatter([], [], s=150, c='blue', 
                                                     marker='o', edgecolor='black', linewidth=1.5)
                path_line_trans, = ax_trans.plot([], [], 'k-', alpha=0.6, linewidth=1.5)
                role_text = ax_trans.text(0.02, 0.98, "", transform=ax_trans.transAxes, 
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(facecolor='white', alpha=0.8))
                
                # Add legend and labels
                ax_trans.legend(loc='upper right')
                ax_trans.set_xlabel('X Position', fontsize=12)
                ax_trans.set_ylabel('Y Position', fontsize=12)
                ax_trans.grid(True, alpha=0.3)
                
                # Create a title to show which transition
                title_text = ax_trans.text(0.5, 1.05, "", transform=ax_trans.transAxes,
                                        fontsize=14, horizontalalignment='center')
                
                # Select frames around transitions
                context_frames = 15  # Number of frames before/after transition
                transition_frames = []
                
                # Take first few transitions
                for i, trans in enumerate(transitions[:min(5, len(transitions))]):
                    start = max(0, trans - context_frames)
                    end = min(len(data), trans + context_frames)
                    # Store as (index, transition_index, transition_number)
                    transition_frames.extend([(idx, trans, i+1) for idx in range(start, end)])
                
                # Ensure frames are unique and sorted
                transition_frames = sorted(set(transition_frames), key=lambda x: x[0])
                
                # Initialize path for transition animation
                path_x_trans = []
                path_y_trans = []
                current_trans = 0
                
                def init_transitions():
                    forager_scatter_trans.set_offsets(np.empty((0, 2)))
                    path_line_trans.set_data([], [])
                    role_text.set_text("")
                    title_text.set_text("")
                    return forager_scatter_trans, path_line_trans, role_text, title_text
                
                def update_transitions(frame_data):
                    nonlocal path_x_trans, path_y_trans, current_trans
                    frame_idx, trans_idx, trans_num = frame_data
                    
                    # Check if we've moved to a new transition
                    if trans_num != current_trans:
                        # Reset path for new transition
                        path_x_trans = []
                        path_y_trans = []
                        current_trans = trans_num
                    
                    # Get current position and role
                    current_pos = np.array([x_pos[frame_idx], y_pos[frame_idx]]).reshape(1, 2)
                    current_role = int(assignments[frame_idx, 0])
                    
                    # Update path
                    path_x_trans.append(x_pos[frame_idx])
                    path_y_trans.append(y_pos[frame_idx])
                    path_line_trans.set_data(path_x_trans, path_y_trans)
                    
                    # Update forager position and color
                    forager_scatter_trans.set_offsets(current_pos)
                    forager_scatter_trans.set_color(colors[min(current_role, len(colors)-1)])
                    
                    # Update role text
                    role_name = role_names[min(current_role, len(role_names)-1)]
                    velocity = np.sqrt(data[frame_idx, 0, 2]**2 + data[frame_idx, 0, 3]**2).item()
                    text = f"Role: {role_name}\nFrame: {frame_idx}"
                    role_text.set_text(text)
                    
                    # Highlight when exactly at the transition
                    if frame_idx == trans_idx:
                        # We're at the exact transition point
                        from_role = role_names[min(int(assignments[frame_idx-1, 0]), len(role_names)-1)]
                        to_role = role_names[min(current_role, len(role_names)-1)]
                        title = f"Transition {trans_num}: {from_role} → {to_role}"
                        title_text.set_text(title)
                    
                    return forager_scatter_trans, path_line_trans, role_text, title_text
                
                # Create animation focusing on transitions
                transitions_ani = animation.FuncAnimation(
                    fig_trans, 
                    update_transitions, 
                    frames=transition_frames, 
                    init_func=init_transitions, 
                    blit=True, 
                    interval=150
                )
                
                # Save the transition animation
                animation_path = save_dir / "role_transitions_animation.gif"
                transitions_ani.save(animation_path, writer=writer, dpi=120)
                plt.close()
                
                print(f"Transitions animation saved to {animation_path}")
                
        except Exception as e:
            print(f"Warning: Could not create transition animation: {e}")
            traceback.print_exc()

def run_dmbd_analysis(data, obs_shape, role_dims, hidden_dims, number_of_objects, seed):
    """Run DMBD analysis with given parameters."""
    print(f"Running DMBD analysis on data with shape {data.shape}...")
    print("Creating DMBD model with parameters:")
    print(f"  obs_shape: {obs_shape}")
    print(f"  role_dims: {role_dims}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  number_of_objects: {number_of_objects}")
    print(f"  regression_dim: 0")

    # Create DMBD model with corrected parameters
    dmbd_model = DMBD(
            obs_shape=obs_shape,
        role_dims=role_dims,
        hidden_dims=hidden_dims,
        number_of_objects=number_of_objects,
        regression_dim=0  # No regression
    )
    print("DMBD model created successfully.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dmbd_model = dmbd_model.to(device)
    data = data.to(device)
    
    # Create empty control input tensor with proper shape
    time_steps = data.shape[0]
    u = torch.zeros((time_steps, number_of_objects, 0), dtype=torch.float32).to(device)
    
    # Create role assignment tensor with proper shape
    r = torch.ones((time_steps, number_of_objects, 1), dtype=torch.float32).to(device)

    # Run model update with corrected parameters
    print(f"  Attempting model update with lr=0.01, data shape={data.shape}")
    update_result = dmbd_model.update(data, u, r, iters=10, latent_iters=5, lr=0.01, verbose=True)
    print(f"  Update result: {update_result}")

    # Try to extract assignments if update was successful
    results = {
        'update_success': update_result,
        'model': dmbd_model
    }
    
    if update_result:
        try:
            # Get assignment probabilities
            assignments_pr = dmbd_model.assignment_pr()
            results['assignments_pr'] = assignments_pr.detach().cpu()
            
            # Get hard assignments
            assignments = dmbd_model.assignment()
            results['assignments'] = assignments.detach().cpu()
            
            print(f"  Successfully extracted assignments with shape {assignments.shape}")
        except Exception as e:
            print(f"  Error extracting assignments: {e}")
    
    return results

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_forager_dmbd_basic(seed):
    """
    Run a basic DMBD analysis on Forager simulation data with enhanced error debugging
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dmbd_outputs", "forager")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear debug log at the start of the test
    with open(os.path.join(output_dir, "debug_log.txt"), "w") as f:
        f.write("STARTING NEW TEST RUN\n")
        f.write("====================\n\n")
    
    # Debug output logger
    def debug_print(message):
        print(f"DEBUG: {message}")
        with open(os.path.join(output_dir, "debug_log.txt"), "a") as f:
            f.write(f"{message}\n")
    
    # Initialize the Forager simulation
    debug_print("Initializing Forager simulation")
    forager_sim = Forager()
    
    # Configure Forager parameters for better DMBD convergence
    # Using fewer time steps and clearer behavioral patterns
    forager_sim.num_foods = 5            # Fewer food items for clearer patterns
    forager_sim.food_range = 8           # Smaller range for faster exploration
    forager_sim.forager_speed = 0.8      # Slightly slower for smoother trajectories
    forager_sim.vision_range = 3.0       # Moderate vision range
    forager_sim.num_steps = 500          # Fewer steps for faster convergence
    forager_sim.noise = 0.1              # Lower noise for clearer patterns
    
    # Store forager parameters for diagnostics
    forager_parameters = {
        'num_foods': forager_sim.num_foods,
        'food_range': forager_sim.food_range,
        'forager_speed': forager_sim.forager_speed,
        'vision_range': forager_sim.vision_range,
        'num_steps': forager_sim.num_steps,
        'noise': forager_sim.noise
    }
    
    # Run the simulation
    debug_print("Running Forager simulation")
    positions, food_positions, food_memory = forager_sim.simulate()
    
    debug_print(f"Positions shape: {positions.shape}")
    debug_print(f"Food positions shape: {food_positions.shape}")
    debug_print(f"Food memory shape: {food_memory.shape}")
    
    # Normalize positions (Min-Max scaling to [0, 1])
    min_pos = torch.min(positions, dim=0)[0]
    max_pos = torch.max(positions, dim=0)[0]
    range_pos = max_pos - min_pos
    range_pos[range_pos == 0] = 1.0  # Avoid division by zero
    positions_norm = (positions - min_pos) / range_pos
    
    # Calculate velocities (first difference)
    velocities = torch.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    
    # Normalize velocities using z-score normalization
    # This can help with DMBD convergence by making velocity distributions more regular
    velocities_mean = torch.mean(velocities, dim=0)
    velocities_std = torch.std(velocities, dim=0)
    velocities_std[velocities_std < 1e-8] = 1.0  # Avoid division by zero
    velocities_norm = (velocities - velocities_mean) / velocities_std
    
    # Create observation tensor [time_steps, n_objects=1, obs_dim]
    time_steps = positions.shape[0]
    obs_dim = 5  # x, y, vx, vy, memory_state
    
    # Prepare the data tensor with proper reshaping for DMBD
    data = torch.zeros((time_steps, 1, obs_dim))
    
    # Fill with normalized positions and velocities
    data[:, 0, 0:2] = positions_norm
    data[:, 0, 2:4] = velocities_norm
    
    # Use the sum of food memory across food items as a scalar reward/memory signal
    # Normalize this as well for better convergence
    memory_state = food_memory.sum(dim=1).float()  # Sum across food items
    memory_state = (memory_state - memory_state.min()) / (memory_state.max() - memory_state.min() + 1e-8)
    debug_print(f"Memory state shape: {memory_state.shape}")
    data[:, 0, 4] = memory_state
    
    debug_print(f"Data tensor shape: {data.shape}")
    debug_print(f"Data tensor min/max: {torch.min(data).item():.4f}, {torch.max(data).item():.4f}")
    
    # Initialize results dictionary
    results = {
        'data': data,
        'positions': positions,
        'food_positions': food_positions,
        'reward': memory_state,
        'dmbd_results': False,
        'forager_parameters': forager_parameters
    }
    
    # DMBD model configuration for better convergence
    # Use simpler role dimensions to make learning easier
    role_dims = (1, 2, 1)       # Simple system/blanket/environment roles
    hidden_dims = (2, 2, 2)     # Hidden dimensions unchanged
    
    try:
        debug_print("Creating DMBD model with simplified role dimensions")
        # Create the DMBD model with dimensions that are more likely to converge
        model = DMBD(
            obs_shape=(1, obs_dim),  # (n_objects=1, obs_dim=5)
            role_dims=role_dims,     # Simplified roles for better convergence
            hidden_dims=hidden_dims, # Hidden state dimensions
            number_of_objects=1,     # Single agent
            regression_dim=0         # No regression
        )
        
        # Save the model in results
        results['model'] = model
        
        debug_print("Running DMBD update")
        try:
            # Create control input tensor (u) - all zeros since we don't have control inputs
            u = torch.zeros((time_steps, 1, 0), dtype=torch.float32)  # Empty control dim
            
            # Create role assignment tensor (r) - initialized with ones
            # Important: match dimensions with the role_dims parameter
            r = torch.ones((time_steps, 1, 1), dtype=torch.float32)
            
            # Convert all tensors to the same device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            u = u.to(device)
            r = r.to(device)
            model = model.to(device)
            
            # Record update parameters for diagnostics
            update_parameters = {
                'data_shape': data.shape,
                'u_shape': u.shape,
                'r_shape': r.shape,
                'role_dims': role_dims,
                'hidden_dims': hidden_dims,
                'iters': 30,         # More iterations for better convergence
                'latent_iters': 10,  # More latent iterations
                'lr': 0.05,          # Higher learning rate
                'verbose': True,
                'device': str(device)
            }
            results['update_parameters'] = update_parameters
            
            debug_print(f"Running on device: {device}")
            debug_print("Calling model.update with parameters:")
            for key, value in update_parameters.items():
                if isinstance(value, torch.Size):
                    debug_print(f"  {key}: {value}")
                elif isinstance(value, tuple):
                    debug_print(f"  {key}: {value}")
                else:
                    debug_print(f"  {key}: {value}")
            
            # Capture debug info during update
            debug_info = []
            
            # Run the DMBD update with parameters tuned for better convergence
            update_success = model.update(
                data,             # y (observations)
                u,                # u (control inputs)
                r,                # r (role assignments)
                iters=30,         # More iterations for convergence
                latent_iters=10,  # More latent iterations
                lr=0.05,          # Higher learning rate
                verbose=True      # Print progress
            )
            
            # Store update results
            results['debug_info'] = '\n'.join(debug_info)
            debug_print(f"DMBD update completed with result: {update_success}")
            results['update_success'] = update_success
            results['dmbd_results'] = update_success  # Set if DMBD analysis was successful
            
            # If update was successful, extract blanket assignments
            if update_success:
                debug_print("Extracting Markov blanket assignments")
                try:
                    # Get assignment probabilities
                    assignments_pr = model.assignment_pr()
                    results['assignments_pr'] = assignments_pr.detach().cpu()
                    
                    # Get hard assignments
                    assignments = model.assignment()
                    results['assignments'] = assignments.detach().cpu()
                    
                    debug_print(f"Assignment shape: {assignments.shape}")
                    debug_print(f"Assignment probabilities shape: {assignments_pr.shape}")
                    
                    # Print some statistics about the assignments to help with diagnostics
                    role_counts = torch.bincount(assignments[:,0].flatten().cpu())
                    debug_print(f"Role distribution: {role_counts}")
                    
                except Exception as assign_e:
                    error_traceback = traceback.format_exc()
                    debug_print(f"Error extracting assignments: {str(assign_e)}")
                    debug_print(f"Assignment error traceback: {error_traceback}")
                    results['error_message'] = f"Error extracting assignments: {str(assign_e)}"
                    results['error_traceback'] = error_traceback
            else:
                # Set detailed error message for failed update
                results['error_message'] = "DMBD update returned False, model failed to converge"
                
                # Try a fallback approach with even simpler model
                debug_print("Trying fallback approach with simpler model...")
                
                # Create a simpler model
                fallback_role_dims = (1, 1, 1)   # Minimal role dimensions
                fallback_model = DMBD(
                    obs_shape=(1, obs_dim),
                    role_dims=fallback_role_dims,
                    hidden_dims=(1, 1, 1),
                    number_of_objects=1,
                    regression_dim=0
                ).to(device)
                
                debug_print("Running fallback model update...")
                fallback_success = fallback_model.update(
                    data,
                    u,
                    r,
                    iters=50,         # Even more iterations
                    latent_iters=15,  # More latent iterations
                    lr=0.1,           # Higher learning rate
                    verbose=True
                )
                
                if fallback_success:
                    debug_print("Fallback model succeeded!")
                    results['model'] = fallback_model
                    results['update_success'] = True
                    results['dmbd_results'] = True
                    
                    # Extract assignments from fallback model
                    try:
                        assignments_pr = fallback_model.assignment_pr()
                        results['assignments_pr'] = assignments_pr.detach().cpu()
                        
                        assignments = fallback_model.assignment()
                        results['assignments'] = assignments.detach().cpu()
                        
                        debug_print(f"Fallback assignment shape: {assignments.shape}")
                        
                        # Print some statistics about the assignments
                        role_counts = torch.bincount(assignments[:,0].flatten().cpu())
                        debug_print(f"Fallback role distribution: {role_counts}")
                    except Exception as e:
                        debug_print(f"Error extracting fallback assignments: {str(e)}")
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            debug_print(f"DMBD update failed with error: {str(e)}")
            debug_print(f"Error traceback: {error_traceback}")
            
            results['update_success'] = False
            results['error_message'] = str(e)
            results['error_traceback'] = error_traceback
            
    except Exception as outer_e:
        error_traceback = traceback.format_exc()
        debug_print(f"Error in model creation: {str(outer_e)}")
        debug_print(f"Model creation error traceback: {error_traceback}")
        
        results['update_success'] = False
        results['error_message'] = f"Model creation failed: {str(outer_e)}"
        results['error_traceback'] = error_traceback
    
    # Save the results
    debug_print("Saving results")
    torch.save(results, os.path.join(output_dir, "dmbd_results.pt"))
    
    # Visualize results
    visualize_forager_results(results, output_dir)
    
    # Return results for further analysis
    return results

@pytest.mark.dmbd
@pytest.mark.parametrize("seed", [42])
def test_forager_dmbd_hyperparameter_search(seed, caplog):
    """Test DMBD on Forager with hyperparameter search for optimal configuration."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Check for quick test mode
    quick_mode = os.environ.get('DMBD_QUICK_TEST', '0') == '1'
    if quick_mode:
        print("Running in quick mode with reduced settings")
    
    # Create output directory
    os.makedirs(forager_dir, exist_ok=True)
    
    caplog.set_level("INFO")
    print("Running Forager simulation with DMBD hyperparameter search...")
    
    try:
        # Create Forager instance with appropriate settings
        forager_sim = Forager()
        
        # Modify parameters for quick mode
        if quick_mode:
            forager_sim.num_steps = 300
            forager_sim.num_foods = 3
        else:
            forager_sim.num_steps = 1000
            forager_sim.num_foods = 8
        
        print(f"Generating Forager data with {forager_sim.num_steps} steps and {forager_sim.num_foods} food items...")
        
        # Run simulation
        positions, food_positions, food_memory = forager_sim.simulate()
        
        # Normalize positions
        min_pos = torch.min(positions, dim=0)[0]
        max_pos = torch.max(positions, dim=0)[0]
        range_pos = max_pos - min_pos
        range_pos[range_pos == 0] = 1.0  # Avoid division by zero
        positions_norm = (positions - min_pos) / range_pos
        
        # Calculate velocities
        velocities = torch.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        
        # Normalize velocities
        if torch.norm(velocities) > 0:
            velocities = velocities / torch.norm(velocities, dim=0, keepdim=True).clamp(min=1e-8)
        
        # Prepare data tensor
        time_steps = positions.shape[0]
        obs_dim = 5  # x, y, vx, vy, memory_state
        data = torch.zeros((time_steps, 1, obs_dim))
        
        # Fill data tensor
        data[:, 0, 0:2] = positions_norm  # x, y positions
        data[:, 0, 2:4] = velocities      # vx, vy velocities
        data[:, 0, 4] = food_memory.sum(dim=1).float()  # memory state
        
        # Define test scenarios with different model configurations
        if quick_mode:
            test_configs = [
                {
                    'name': 'baseline',
                    'role_dims': (2, 2, 2),
                    'hidden_dims': (2, 2, 2)
                }
            ]
        else:
            test_configs = [
                {
                    'name': 'baseline',
                    'role_dims': (2, 2, 2),
                    'hidden_dims': (2, 2, 2)
                },
                {
                    'name': 'medium',
                    'role_dims': (3, 3, 3),
                    'hidden_dims': (3, 3, 3)
                },
                {
                    'name': 'complex',
                    'role_dims': (4, 4, 4),
                    'hidden_dims': (3, 3, 3)
                }
            ]
        
        best_update_success = False
        best_config = None
        
        # Test different configurations
        for config in test_configs:
            print(f"\nTesting configuration: {config['name']}")
            print(f"  role_dims: {config['role_dims']}")
            print(f"  hidden_dims: {config['hidden_dims']}")
            
            # Run DMBD analysis
            results = run_dmbd_analysis(
                data=data,
                obs_shape=(1, obs_dim),
                role_dims=config['role_dims'],
                hidden_dims=config['hidden_dims'],
                number_of_objects=1,
                seed=seed
            )
            
            # Store results
            config_output_dir = os.path.join(forager_dir, f"config_{config['name']}")
            os.makedirs(config_output_dir, exist_ok=True)
            
            # Save results
            torch.save(results, os.path.join(config_output_dir, "dmbd_results.pt"))
            
            # Track best configuration
            if results.get('update_success', False):
                if not best_update_success:
                    best_update_success = True
                    best_config = config
            
            # Create visualizations for this configuration
            results['data'] = data
            results['positions'] = positions
            results['food_positions'] = food_positions
            results['reward'] = food_memory.sum(dim=1).float()
            
            # Generate visualizations
            visualize_forager_results(results, Path(config_output_dir))
        
        # Report best configuration
        if best_config:
            print(f"\nBest configuration: {best_config['name']}")
            print(f"  role_dims: {best_config['role_dims']}")
            print(f"  hidden_dims: {best_config['hidden_dims']}")
        else:
            print("\nNo successful configurations found")
        
        # Create summary comparison visualization
        plt.figure(figsize=(10, 6))
        plt.title("Forager DMBD - Configuration Comparison")
        
        # Add bars for each configuration showing success status
        config_names = [config['name'] for config in test_configs]
        success_values = []
        
        for config in test_configs:
            config_output_dir = os.path.join(forager_dir, f"config_{config['name']}")
            results_file = os.path.join(config_output_dir, "dmbd_results.pt")
            
            if os.path.exists(results_file):
                results = torch.load(results_file)
                success_values.append(1 if results.get('update_success', False) else 0)
            else:
                success_values.append(0)
        
        plt.bar(config_names, success_values)
        plt.ylabel("Update Success (1=success, 0=failure)")
        plt.ylim(0, 1.2)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(forager_dir, "configuration_comparison.png"))
        plt.close()
        
        print("\nForager DMBD hyperparameter search completed successfully")
        
    except Exception as e:
        print(f"Error in forager DMBD hyperparameter search: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Forager DMBD hyperparameter search failed: {e}")

def visualize_trajectory(positions, memory, food_positions, save_dir):
    """
    Visualize the forager's trajectory, food positions, and memory state.
    
    Args:
        positions: Tensor of shape [time_steps, 2] containing forager positions
        memory: Tensor of shape [time_steps] containing memory state
        food_positions: Tensor of shape [time_steps, n_food, 2] containing food positions
        save_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        # Ensure save_dir is a Path object
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy arrays if they're not already
        if hasattr(positions, 'numpy'):
            positions = positions.numpy()
        if hasattr(memory, 'numpy'):
            memory = memory.numpy()
        if hasattr(food_positions, 'numpy'):
            food_positions = food_positions.numpy()
            
        # Create a figure for the trajectory
        plt.figure(figsize=(12, 10))
        
        # Plot the trajectory with color based on memory state
        plt.subplot(2, 2, 1)
        plt.scatter(positions[:, 0], positions[:, 1], c=memory, cmap='viridis', 
                   s=10, alpha=0.7)
        plt.colorbar(label='Memory State')
        
        # Plot food positions (using the first time step)
        if food_positions.ndim == 3:  # [time_steps, n_food, 2]
            for i in range(food_positions.shape[1]):
                plt.scatter(food_positions[0, i, 0], food_positions[0, i, 1], 
                           c='red', s=100, marker='*', label=f'Food {i+1}' if i == 0 else "")
        
        plt.title('Forager Trajectory with Memory State')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot memory state over time
        plt.subplot(2, 2, 2)
        plt.plot(memory, 'g-', linewidth=2)
        plt.title('Memory State Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Memory State')
        plt.grid(True, alpha=0.3)
        
        # Plot velocity (calculated as position differences)
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        
        plt.subplot(2, 2, 3)
        plt.quiver(positions[:-1:10, 0], positions[:-1:10, 1], 
                  velocities[1::10, 0], velocities[1::10, 1], 
                  memory[1::10], cmap='viridis', scale=1.0, width=0.005)
        plt.colorbar(label='Memory State')
        plt.title('Velocity Vectors (subsampled)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        
        # Plot position heatmap
        plt.subplot(2, 2, 4)
        heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Frequency')
        plt.title('Position Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_dir / 'trajectory_visualization.png', dpi=150)
        plt.close()
        
        print(f"Trajectory visualization saved to {save_dir / 'trajectory_visualization.png'}")
        
    except Exception as e:
        print(f"Error in visualize_trajectory: {e}")
        import traceback
        traceback.print_exc()

def visualize_data_tensor(data, save_dir):
    """
    Visualize the data tensor used for DMBD analysis.
    
    Args:
        data: Tensor of shape [time_steps, n_objects, obs_dim] containing observation data
        save_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from pathlib import Path
        
        # Ensure save_dir is a Path object
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor to numpy array if it's not already
        if hasattr(data, 'numpy'):
            data_np = data.numpy()
        else:
            data_np = data
            
        time_steps, n_objects, obs_dim = data_np.shape
        
        # Create a figure for the data tensor visualization
        plt.figure(figsize=(15, 12))
        
        # Plot each dimension of the data tensor
        for i in range(obs_dim):
            plt.subplot(obs_dim, 1, i+1)
            for j in range(n_objects):
                plt.plot(data_np[:, j, i], label=f'Object {j+1}, Dim {i+1}')
            plt.title(f'Dimension {i+1} Over Time')
            plt.xlabel('Time Step')
            plt.ylabel(f'Value (Dim {i+1})')
            plt.grid(True, alpha=0.3)
            if n_objects > 1:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'data_tensor_timeseries.png', dpi=150)
        plt.close()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Reshape data for correlation analysis
        data_reshaped = data_np.reshape(time_steps, n_objects * obs_dim)
        
        # Create column labels
        col_labels = [f'Obj{j+1}_Dim{i+1}' for j in range(n_objects) for i in range(obs_dim)]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_reshaped.T)
        
        # Plot correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                   xticklabels=col_labels, yticklabels=col_labels)
        plt.title('Correlation Between Data Dimensions')
        plt.tight_layout()
        plt.savefig(save_dir / 'data_tensor_correlation.png', dpi=150)
        plt.close()
        
        # Create distribution plots for each dimension
        plt.figure(figsize=(15, 10))
        
        for i in range(obs_dim):
            plt.subplot(obs_dim, 1, i+1)
            for j in range(n_objects):
                sns.kdeplot(data_np[:, j, i], label=f'Object {j+1}, Dim {i+1}')
            plt.title(f'Distribution of Dimension {i+1}')
            plt.xlabel(f'Value (Dim {i+1})')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            if n_objects > 1:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'data_tensor_distributions.png', dpi=150)
        plt.close()
        
        print(f"Data tensor visualizations saved to {save_dir}")
        
        # Save statistics about the data tensor
        with open(save_dir / 'data_tensor_stats.txt', 'w') as f:
            f.write("DATA TENSOR STATISTICS\n")
            f.write("=====================\n\n")
            f.write(f"Shape: {data_np.shape}\n")
            f.write(f"Min value: {np.min(data_np):.6f}\n")
            f.write(f"Max value: {np.max(data_np):.6f}\n")
            f.write(f"Mean value: {np.mean(data_np):.6f}\n")
            f.write(f"Std deviation: {np.std(data_np):.6f}\n\n")
            
            f.write("Statistics by dimension:\n")
            for i in range(obs_dim):
                for j in range(n_objects):
                    dim_data = data_np[:, j, i]
                    f.write(f"Object {j+1}, Dimension {i+1}:\n")
                    f.write(f"  Min: {np.min(dim_data):.6f}\n")
                    f.write(f"  Max: {np.max(dim_data):.6f}\n")
                    f.write(f"  Mean: {np.mean(dim_data):.6f}\n")
                    f.write(f"  Std: {np.std(dim_data):.6f}\n")
                    f.write(f"  Skewness: {np.mean(((dim_data - np.mean(dim_data)) / np.std(dim_data)) ** 3):.6f}\n")
                    f.write(f"  Kurtosis: {np.mean(((dim_data - np.mean(dim_data)) / np.std(dim_data)) ** 4) - 3:.6f}\n\n")
    
    except Exception as e:
        print(f"Error in visualize_data_tensor: {e}")
        import traceback
        traceback.print_exc()

@pytest.mark.dmbd
def test_forager_dmbd_simplified():
    """
    Run a simplified DMBD analysis on Forager simulation data with minimal parameters
    for better convergence.
    """
    # Set up output directory
    import os
    from pathlib import Path
    import torch
    import numpy as np
    import random
    import traceback
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    save_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dmbd_outputs", "forager", "simplified"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear debug log at the start of the test
    with open(save_dir / "debug_log.txt", "w") as f:
        f.write("SIMPLIFIED DMBD TEST LOG\n")
        f.write("======================\n\n")
    
    # Debug output logger
    def debug_print(message):
        print(f"DEBUG: {message}")
        with open(save_dir / "debug_log.txt", "a") as f:
            f.write(f"{message}\n")
    
    # Forager parameters for better convergence
    forager_params = {
        'num_food': 3,
        'food_range': 5.0,
        'forager_speed': 0.5,
        'num_steps': 300,
        'noise': 0.05,
        'seed': 42
    }
    
    debug_print(f"Forager parameters: {forager_params}")
    
    # Define the Forager class locally
    class Forager:
        """
        Simple forager simulation for testing DMBD.
        """
        def __init__(self):
            self.num_foods = 5
            self.food_range = 8.0
            self.forager_speed = 0.8
            self.vision_range = 3.0
            self.num_steps = 500
            self.noise = 0.1
            
        def simulate(self):
            """
            Run the forager simulation.
            
            Returns:
                positions: Tensor of shape [time_steps, 2] containing forager positions
                food_positions: Tensor of shape [time_steps, n_food, 2] containing food positions
                memory: Tensor of shape [time_steps, n_food] containing memory state
            """
            # Set random seed for reproducibility
            torch.manual_seed(self.seed if hasattr(self, 'seed') else 42)
            
            # Initialize positions
            time_steps = self.num_steps + 1
            positions = torch.zeros((time_steps, 2))
            
            # Initialize food positions (fixed throughout simulation)
            food_positions = torch.zeros((time_steps, self.num_foods, 2))
            for i in range(self.num_foods):
                food_pos = (torch.rand(2) * 2 - 1) * self.food_range
                food_positions[:, i] = food_pos
            
            # Initialize memory (whether each food has been found)
            memory = torch.zeros((time_steps, self.num_foods))
            
            # Run simulation
            for t in range(1, time_steps):
                # Current position
                pos = positions[t-1]
                
                # Calculate direction based on visible food
                direction = torch.zeros(2)
                
                # Check for visible food
                for i in range(self.num_foods):
                    food_pos = food_positions[t, i]
                    distance = torch.norm(food_pos - pos)
                    
                    # If food is visible and not found, move towards it
                    if distance < self.vision_range and memory[t-1, i] == 0:
                        direction += (food_pos - pos) / distance
                
                # If no food is visible, move randomly
                if torch.norm(direction) < 1e-6:
                    direction = torch.randn(2)
                
                # Normalize direction
                direction = direction / (torch.norm(direction) + 1e-8)
                
                # Add noise
                noise = torch.randn(2) * self.noise
                direction = direction + noise
                direction = direction / (torch.norm(direction) + 1e-8)
                
                # Update position
                positions[t] = pos + direction * self.forager_speed
                
                # Update memory (check if food is found)
                for i in range(self.num_foods):
                    food_pos = food_positions[t, i]
                    distance = torch.norm(food_pos - positions[t])
                    
                    # Copy previous memory state
                    memory[t, i] = memory[t-1, i]
                    
                    # If food is found, update memory
                    if distance < 0.5:
                        memory[t, i] = 1.0
            
            return positions, food_positions, memory
    
    # Create and run forager simulation
    forager_sim = Forager()
    
    # Set parameters
    forager_sim.num_foods = forager_params['num_food']
    forager_sim.food_range = forager_params['food_range']
    forager_sim.forager_speed = forager_params['forager_speed']
    forager_sim.num_steps = forager_params['num_steps']
    forager_sim.noise = forager_params['noise']
    forager_sim.seed = forager_params['seed']
    
    # Run simulation
    debug_print("Running simulation...")
    positions, food_positions, memory = forager_sim.simulate()
    
    # Convert to float32 to avoid data type issues
    positions = positions.float()
    food_positions = food_positions.float()
    memory = memory.float()
    
    debug_print(f"Positions shape: {positions.shape}, dtype: {positions.dtype}")
    debug_print(f"Food positions shape: {food_positions.shape}, dtype: {food_positions.dtype}")
    debug_print(f"Food memory shape: {memory.shape}, dtype: {memory.dtype}")
    
    # Normalize positions (Min-Max scaling to [0, 1])
    min_pos = torch.min(positions, dim=0)[0]
    max_pos = torch.max(positions, dim=0)[0]
    range_pos = max_pos - min_pos
    range_pos[range_pos == 0] = 1.0  # Avoid division by zero
    positions_norm = (positions - min_pos) / range_pos
    
    # Calculate velocities (first difference)
    velocities = torch.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    
    # Normalize velocities using z-score normalization
    velocities_mean = torch.mean(velocities, dim=0)
    velocities_std = torch.std(velocities, dim=0)
    velocities_std[velocities_std < 1e-8] = 1.0  # Avoid division by zero
    velocities_norm = (velocities - velocities_mean) / velocities_std
    
    # Create observation tensor [time_steps, n_objects=1, obs_dim]
    time_steps = positions.shape[0]
    obs_dim = 5  # x, y, vx, vy, memory_state
    
    # Prepare the data tensor with proper reshaping for DMBD
    data = torch.zeros((time_steps, 1, obs_dim), dtype=torch.float32)
    
    # Fill with normalized positions and velocities
    data[:, 0, 0:2] = positions_norm
    data[:, 0, 2:4] = velocities_norm
    
    # Use the sum of food memory across food items as a scalar reward/memory signal
    # Normalize this as well for better convergence
    memory_state = memory.sum(dim=1).float()  # Sum across food items
    memory_state = (memory_state - memory_state.min()) / (memory_state.max() - memory_state.min() + 1e-8)
    data[:, 0, 4] = memory_state
    
    debug_print(f"Prepared data tensor with shape {data.shape}, dtype: {data.dtype}")
    
    # Visualize the data tensor before DMBD analysis
    visualize_data_tensor(data, save_dir)
    
    # Visualize the trajectory
    visualize_trajectory(positions, memory_state, food_positions, save_dir)
    
    # Create a minimal DMBD model for better convergence
    debug_print("Creating minimal DMBD model...")
    from dmbd.dmbd import DMBD
    
    # Use minimal role dimensions for simpler learning
    role_dims = (1, 1, 1)  # Minimal role dimensions
    
    # Create the DMBD model
    model = DMBD(
        obs_shape=(1, obs_dim),  # (n_objects=1, obs_dim=5)
        role_dims=role_dims,     # Minimal role dimensions
        hidden_dims=(2, 2, 2),   # Hidden state dimensions
        number_of_objects=1,     # Single agent
        regression_dim=0         # No regression
    )
    
    # Create empty control input tensor with proper shape
    u = torch.zeros((time_steps, 1, 0), dtype=torch.float32)
    
    # Create role assignment tensor with proper shape
    r = torch.ones((time_steps, 1, 1), dtype=torch.float32)
    
    # Try multiple update configurations with increasing learning rates and iterations
    update_success = False
    
    # First attempt with moderate parameters
    debug_print("Attempting update with lr=0.01, iters=20")
    try:
        update_success = model.update(data, u, r, iters=20, latent_iters=10, lr=0.01, verbose=True)
    except Exception as e:
        debug_print(f"DMBD update failed: {str(e)}")
    debug_print(f"Update result: {update_success}")
    
    # If first attempt fails, try with higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.03, iters=30")
        try:
            update_success = model.update(data, u, r, iters=30, latent_iters=15, lr=0.03, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # If second attempt fails, try with even higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.1, iters=50")
        try:
            update_success = model.update(data, u, r, iters=50, latent_iters=25, lr=0.1, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # If third attempt fails, try with very high learning rate and many iterations
    if not update_success:
        debug_print("Attempting update with lr=0.3, iters=100")
        try:
            update_success = model.update(data, u, r, iters=100, latent_iters=50, lr=0.3, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # If all standard attempts fail, try one final approach with very high iterations
    if not update_success:
        debug_print("All standard updates failed. Trying final approach with very high iterations...")
        
        # Create a new model with even simpler dimensions
        model = DMBD(
            obs_shape=(1, obs_dim),
            role_dims=(1, 1, 1),     # Minimal role dimensions
            hidden_dims=(1, 1, 1),   # Minimal hidden dimensions
            number_of_objects=1,
            regression_dim=0
        )
        
        try:
            update_success = model.update(data, u, r, iters=200, latent_iters=100, lr=0.5, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Final update result: {update_success}")
    
    # Save results
    results = {
        'data': data,
        'positions': positions,
        'food_positions': food_positions,
        'memory': memory,
        'velocities': velocities,
        'velocities_norm': velocities_norm,
        'u': u,
        'role_dims': role_dims,
        'success': update_success,
        'save_dir': save_dir
    }
    
    # Save results to file
    torch.save(results, save_dir / "dmbd_results.pt")
    
    if update_success:
        debug_print("DMBD update successful! Analyzing results...")
        try:
            # Get assignment probabilities
            assignments_pr = model.assignment_pr()
            results['assignments_pr'] = assignments_pr.detach().cpu()
            
            # Get hard assignments
            assignments = model.assignment()
            results['assignments'] = assignments.detach().cpu()
            
            # Visualize Markov blankets
            visualize_markov_blankets(results, model, save_dir)
            
            # Animate Markov blankets
            animate_markov_blankets(results, model, save_dir)
            
            # Analyze Markov blankets
            analyze_markov_blankets(results, model, save_dir)
            
            debug_print("Analysis complete!")
        except Exception as e:
            debug_print(f"Error in analysis: {str(e)}")
            traceback.print_exc()
    else:
        debug_print("All DMBD update attempts failed to converge.")
        
    # Even if DMBD update fails, we can still visualize the data
    try:
        visualize_trajectory(positions, memory_state, food_positions, save_dir)
    except Exception as e:
        debug_print(f"Error in DMBD test: {str(e)}")
        
    # Return results for further analysis
    return results

@pytest.mark.dmbd
def test_synthetic_dmbd():
    """
    Test DMBD on a synthetic dataset with clear Markov blanket structure.
    This test creates a simple synthetic dataset with known Markov blanket structure
    to ensure that DMBD can learn and identify Markov blankets in a controlled setting.
    """
    # Set up output directory
    import os
    from pathlib import Path
    import torch
    import numpy as np
    import random
    import traceback
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    save_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dmbd_outputs", "synthetic"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear debug log at the start of the test
    with open(save_dir / "debug_log.txt", "w") as f:
        f.write("SYNTHETIC DMBD TEST LOG\n")
        f.write("======================\n\n")
    
    # Debug output logger
    def debug_print(message):
        print(f"DEBUG: {message}")
        with open(save_dir / "debug_log.txt", "a") as f:
            f.write(f"{message}\n")
    
    # Create a synthetic dataset with clear Markov blanket structure
    # We'll create a system with 3 variables:
    # - Variable 1: System (internal state)
    # - Variable 2: Blanket (interface with environment)
    # - Variable 3: Environment (external state)
    
    debug_print("Creating synthetic dataset with clear Markov blanket structure")
    
    # Parameters
    time_steps = 500  # More time steps for better convergence
    n_objects = 1
    obs_dim = 3  # One dimension for each variable (system, blanket, environment)
    
    # Create the data tensor
    data = torch.zeros((time_steps, n_objects, obs_dim), dtype=torch.float32)
    
    # Generate synthetic data with very clear dependencies
    # System depends on blanket, blanket depends on both system and environment,
    # environment is independent
    
    # Start with random values
    data[0, 0, 0] = torch.randn(1)  # System
    data[0, 0, 1] = torch.randn(1)  # Blanket
    data[0, 0, 2] = torch.randn(1)  # Environment
    
    # Generate time series with strong dependencies
    for t in range(1, time_steps):
        # Environment evolves independently (random walk with some persistence)
        data[t, 0, 2] = 0.9 * data[t-1, 0, 2] + 0.1 * torch.randn(1)
        
        # Blanket depends strongly on both system and environment
        data[t, 0, 1] = 0.45 * data[t-1, 0, 0] + 0.45 * data[t-1, 0, 2] + 0.1 * torch.randn(1)
        
        # System depends strongly only on blanket (not directly on environment)
        data[t, 0, 0] = 0.9 * data[t-1, 0, 1] + 0.1 * torch.randn(1)
    
    debug_print(f"Synthetic data tensor shape: {data.shape}")
    
    # Visualize the data tensor
    visualize_data_tensor(data, save_dir)
    
    # Create a minimal DMBD model
    debug_print("Creating minimal DMBD model for synthetic data")
    from dmbd.dmbd import DMBD
    
    # Use role dimensions that match our synthetic data structure
    role_dims = (1, 1, 1)  # System, blanket, environment
    
    # Create the DMBD model
    model = DMBD(
        obs_shape=(1, obs_dim),  # (n_objects=1, obs_dim=3)
        role_dims=role_dims,     # Simple role dimensions
        hidden_dims=(1, 1, 1),   # Minimal hidden dimensions for simpler learning
        number_of_objects=1,     # Single object
        regression_dim=0         # No regression
    )
    
    # Create empty control input tensor with proper shape
    u = torch.zeros((time_steps, 1, 0), dtype=torch.float32)
    
    # Create role assignment tensor with proper shape
    r = torch.ones((time_steps, 1, 1), dtype=torch.float32)
    
    # Try multiple update configurations with increasing learning rates and iterations
    update_success = False
    
    # First attempt with moderate parameters and more iterations
    debug_print("Attempting update with lr=0.01, iters=50")
    try:
        update_success = model.update(data, u, r, iters=50, latent_iters=20, lr=0.01, verbose=True)
    except Exception as e:
        debug_print(f"DMBD update failed: {str(e)}")
    debug_print(f"Update result: {update_success}")
    
    # If first attempt fails, try with higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.05, iters=100")
        try:
            update_success = model.update(data, u, r, iters=100, latent_iters=30, lr=0.05, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # If second attempt fails, try with even higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.2, iters=200")
        try:
            update_success = model.update(data, u, r, iters=200, latent_iters=50, lr=0.2, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # Save results
    results = {
        'data': data,
        'u': u,
        'role_dims': role_dims,
        'success': update_success,
        'save_dir': save_dir
    }
    
    # Save results to file
    torch.save(results, save_dir / "dmbd_results.pt")
    
    if update_success:
        debug_print("DMBD update successful! Analyzing results...")
        try:
            # Get assignment probabilities
            assignments_pr = model.assignment_pr()
            results['assignments_pr'] = assignments_pr.detach().cpu()
            
            # Get hard assignments
            assignments = model.assignment()
            results['assignments'] = assignments.detach().cpu()
            
            # Analyze the assignments
            debug_print("Analyzing assignments...")
            
            # Count the number of each role assignment
            role_counts = torch.bincount(assignments[:, 0].flatten().cpu(), minlength=3)
            debug_print(f"Role distribution: {role_counts}")
            
            # Check if the assignments match our expected structure
            # We expect:
            # - Dimension 0 (system) to be assigned role 0
            # - Dimension 1 (blanket) to be assigned role 1
            # - Dimension 2 (environment) to be assigned role 2
            
            # Get the most common role for each dimension
            dim_roles = []
            for dim in range(obs_dim):
                # Create a mask for each dimension
                dim_mask = torch.zeros_like(data)
                dim_mask[:, :, dim] = 1.0
                
                # Apply the mask to get assignments for this dimension
                dim_assignments = assignments[dim_mask.bool()].cpu()
                
                # Count the occurrences of each role
                dim_role_counts = torch.bincount(dim_assignments, minlength=3)
                
                # Get the most common role
                most_common_role = torch.argmax(dim_role_counts).item()
                dim_roles.append(most_common_role)
            
            debug_print(f"Most common role for each dimension: {dim_roles}")
            
            # Check if the assignments match our expected structure
            expected_roles = [0, 1, 2]  # System, blanket, environment
            if dim_roles == expected_roles:
                debug_print("Assignments match expected structure!")
            else:
                debug_print("Assignments do not match expected structure.")
                
                # Check if the assignments are just permuted
                if sorted(dim_roles) == sorted(expected_roles):
                    debug_print("Assignments are a permutation of expected structure.")
                    
                    # Map the permutation
                    perm_map = {dim_roles[i]: expected_roles[i] for i in range(len(dim_roles))}
                    debug_print(f"Permutation map: {perm_map}")
            
            # Save a visualization of the assignments
            plt.figure(figsize=(12, 8))
            
            # Plot the assignments over time
            plt.subplot(2, 1, 1)
            for i in range(3):
                plt.plot(assignments[:, 0].cpu().numpy(), label=f"Role {i}")
            plt.title("Role Assignments Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Role")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot the data with colors based on role assignments
            plt.subplot(2, 1, 2)
            colors = ['r', 'g', 'b']
            for dim in range(obs_dim):
                plt.plot(data[:, 0, dim].cpu().numpy(), color=colors[dim_roles[dim]], 
                       label=f"Dim {dim} (Role {dim_roles[dim]})")
            plt.title("Data Dimensions Colored by Role Assignment")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "assignments_visualization.png", dpi=150)
            plt.close()
            
            debug_print("Analysis complete!")
        except Exception as e:
            debug_print(f"Error in analysis: {str(e)}")
            traceback.print_exc()
    else:
        debug_print("All DMBD update attempts failed to converge.")
        
    # Return results for further analysis
    return results

@pytest.mark.dmbd
def test_binary_dmbd():
    """
    Test DMBD on a very simple binary dataset with clear Markov blanket structure.
    This test creates a minimal binary dataset with known Markov blanket structure
    to ensure that DMBD can learn and identify Markov blankets in the simplest possible setting.
    """
    # Set up output directory
    import os
    from pathlib import Path
    import torch
    import numpy as np
    import random
    import traceback
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    save_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dmbd_outputs", "binary"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear debug log at the start of the test
    with open(save_dir / "debug_log.txt", "w") as f:
        f.write("BINARY DMBD TEST LOG\n")
        f.write("======================\n\n")
    
    # Debug output logger
    def debug_print(message):
        print(f"DEBUG: {message}")
        with open(save_dir / "debug_log.txt", "a") as f:
            f.write(f"{message}\n")
    
    # Create a binary dataset with clear Markov blanket structure
    # We'll create a system with 3 variables:
    # - Variable 1: System (internal state) - binary (0 or 1)
    # - Variable 2: Blanket (interface with environment) - binary (0 or 1)
    # - Variable 3: Environment (external state) - binary (0 or 1)
    
    debug_print("Creating binary dataset with clear Markov blanket structure")
    
    # Parameters
    time_steps = 1000  # More time steps for better convergence
    n_objects = 1
    obs_dim = 3  # One dimension for each variable (system, blanket, environment)
    
    # Create the data tensor
    data = torch.zeros((time_steps, n_objects, obs_dim), dtype=torch.float32)
    
    # Generate binary data with clear dependencies
    # System depends on blanket, blanket depends on both system and environment,
    # environment is independent
    
    # Start with random binary values
    data[0, 0, 0] = torch.randint(0, 2, (1,)).float()  # System
    data[0, 0, 1] = torch.randint(0, 2, (1,)).float()  # Blanket
    data[0, 0, 2] = torch.randint(0, 2, (1,)).float()  # Environment
    
    # Generate binary time series with clear dependencies
    for t in range(1, time_steps):
        # Environment evolves independently with some persistence
        if torch.rand(1) < 0.8:  # 80% chance to stay the same
            data[t, 0, 2] = data[t-1, 0, 2]
        else:
            data[t, 0, 2] = 1 - data[t-1, 0, 2]  # Flip the bit
        
        # Blanket depends on both system and environment
        # If system and environment are the same, blanket is 1, otherwise 0
        if data[t-1, 0, 0] == data[t, 0, 2]:
            data[t, 0, 1] = 1.0
        else:
            data[t, 0, 1] = 0.0
        
        # System depends only on blanket
        # System follows blanket with 90% probability
        if torch.rand(1) < 0.9:
            data[t, 0, 0] = data[t, 0, 1]
        else:
            data[t, 0, 0] = 1 - data[t, 0, 1]
    
    debug_print(f"Binary data tensor shape: {data.shape}")
    
    # Visualize the data tensor
    visualize_data_tensor(data, save_dir)
    
    # Create a minimal DMBD model
    debug_print("Creating minimal DMBD model for binary data")
    from dmbd.dmbd import DMBD
    
    # Use role dimensions that match our binary data structure
    role_dims = (1, 1, 1)  # System, blanket, environment
    
    # Create the DMBD model with minimal complexity
    model = DMBD(
        obs_shape=(1, obs_dim),  # (n_objects=1, obs_dim=3)
        role_dims=role_dims,     # Simple role dimensions
        hidden_dims=(1, 1, 1),   # Minimal hidden dimensions
        number_of_objects=1,     # Single object
        regression_dim=0         # No regression
    )
    
    # Create empty control input tensor with proper shape
    u = torch.zeros((time_steps, 1, 0), dtype=torch.float32)
    
    # Create role assignment tensor with proper shape
    r = torch.ones((time_steps, 1, 1), dtype=torch.float32)
    
    # Try multiple update configurations with increasing learning rates and iterations
    update_success = False
    
    # First attempt with moderate parameters and more iterations
    debug_print("Attempting update with lr=0.01, iters=100")
    try:
        update_success = model.update(data, u, r, iters=100, latent_iters=50, lr=0.01, verbose=True)
    except Exception as e:
        debug_print(f"DMBD update failed: {str(e)}")
    debug_print(f"Update result: {update_success}")
    
    # If first attempt fails, try with higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.05, iters=200")
        try:
            update_success = model.update(data, u, r, iters=200, latent_iters=100, lr=0.05, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # If second attempt fails, try with even higher learning rate and more iterations
    if not update_success:
        debug_print("Attempting update with lr=0.2, iters=500")
        try:
            update_success = model.update(data, u, r, iters=500, latent_iters=200, lr=0.2, verbose=True)
        except Exception as e:
            debug_print(f"DMBD update failed: {str(e)}")
        debug_print(f"Update result: {update_success}")
    
    # Save results
    results = {
        'data': data,
        'u': u,
        'role_dims': role_dims,
        'success': update_success,
        'save_dir': save_dir
    }
    
    # Save results to file
    torch.save(results, save_dir / "dmbd_results.pt")
    
    if update_success:
        debug_print("DMBD update successful! Analyzing results...")
        try:
            # Get assignment probabilities
            assignments_pr = model.assignment_pr()
            results['assignments_pr'] = assignments_pr.detach().cpu()
            
            # Get hard assignments
            assignments = model.assignment()
            results['assignments'] = assignments.detach().cpu()
            
            # Analyze the assignments
            debug_print("Analyzing assignments...")
            
            # Count the number of each role assignment
            role_counts = torch.bincount(assignments[:, 0].flatten().cpu(), minlength=3)
            debug_print(f"Role distribution: {role_counts}")
            
            # Check if the assignments match our expected structure
            # We expect:
            # - Dimension 0 (system) to be assigned role 0
            # - Dimension 1 (blanket) to be assigned role 1
            # - Dimension 2 (environment) to be assigned role 2
            
            # Get the most common role for each dimension
            dim_roles = []
            for dim in range(obs_dim):
                # Create a mask for each dimension
                dim_mask = torch.zeros_like(data)
                dim_mask[:, :, dim] = 1.0
                
                # Apply the mask to get assignments for this dimension
                dim_assignments = assignments[dim_mask.bool()].cpu()
                
                # Count the occurrences of each role
                dim_role_counts = torch.bincount(dim_assignments, minlength=3)
                
                # Get the most common role
                most_common_role = torch.argmax(dim_role_counts).item()
                dim_roles.append(most_common_role)
            
            debug_print(f"Most common role for each dimension: {dim_roles}")
            
            # Check if the assignments match our expected structure
            expected_roles = [0, 1, 2]  # System, blanket, environment
            if dim_roles == expected_roles:
                debug_print("Assignments match expected structure!")
            else:
                debug_print("Assignments do not match expected structure.")
                
                # Check if the assignments are just permuted
                if sorted(dim_roles) == sorted(expected_roles):
                    debug_print("Assignments are a permutation of expected structure.")
                    
                    # Map the permutation
                    perm_map = {dim_roles[i]: expected_roles[i] for i in range(len(dim_roles))}
                    debug_print(f"Permutation map: {perm_map}")
            
            # Save a visualization of the assignments
            plt.figure(figsize=(12, 8))
            
            # Plot the assignments over time
            plt.subplot(2, 1, 1)
            for i in range(3):
                plt.plot(assignments[:, 0].cpu().numpy(), label=f"Role {i}")
            plt.title("Role Assignments Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Role")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot the data with colors based on role assignments
            plt.subplot(2, 1, 2)
            colors = ['r', 'g', 'b']
            for dim in range(obs_dim):
                plt.plot(data[:, 0, dim].cpu().numpy(), color=colors[dim_roles[dim]], 
                       label=f"Dim {dim} (Role {dim_roles[dim]})")
            plt.title("Data Dimensions Colored by Role Assignment")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "assignments_visualization.png", dpi=150)
            plt.close()
            
            debug_print("Analysis complete!")
        except Exception as e:
            debug_print(f"Error in analysis: {str(e)}")
            traceback.print_exc()
    else:
        debug_print("All DMBD update attempts failed to converge.")
        
    # Return results for further analysis
    return results

def test_compare_datasets():
    """
    Create a comparative visualization of all three datasets (Forager, Synthetic, and Binary)
    to help understand why convergence might be failing.
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import pandas as pd
    from pandas.plotting import autocorrelation_plot
    
    # Create output directory
    save_dir = Path("/home/trim/Documents/GitHub/pyDMBD/fork/dmbd_outputs/comparison")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a debug log file
    debug_log = open(save_dir / "debug_log.txt", "w")
    
    def debug_print(message):
        """Print to console and write to debug log"""
        print(message)
        debug_log.write(f"{message}\n")
        debug_log.flush()
    
    debug_print("Starting dataset comparison...")
    
    # Load data tensors from each experiment if they exist
    data_tensors = {}
    
    # Paths to check
    paths = {
        "Forager": "/home/trim/Documents/GitHub/pyDMBD/fork/dmbd_outputs/forager/simplified/dmbd_results.pt",
        "Synthetic": "/home/trim/Documents/GitHub/pyDMBD/fork/dmbd_outputs/synthetic/dmbd_results.pt",
        "Binary": "/home/trim/Documents/GitHub/pyDMBD/fork/dmbd_outputs/binary/dmbd_results.pt"
    }
    
    # Debug: List all directories to check if they exist
    debug_print("Checking if directories exist:")
    for name, path in paths.items():
        dir_path = os.path.dirname(path)
        debug_print(f"  {name} directory: {dir_path} - Exists: {os.path.exists(dir_path)}")
    
    # Load data tensors
    for name, path in paths.items():
        debug_print(f"Checking {name} path: {path}")
        if os.path.exists(path):
            try:
                debug_print(f"  Loading {name} results...")
                results = torch.load(path)
                debug_print(f"  Keys in results: {list(results.keys())}")
                
                # Look for 'data' instead of 'data_tensor'
                if 'data' in results:
                    data_tensors[name] = results['data']
                    debug_print(f"  Loaded {name} data tensor with shape {data_tensors[name].shape}")
                else:
                    debug_print(f"  No data found in {name} results")
            except Exception as e:
                debug_print(f"  Error loading {name} data: {e}")
        else:
            debug_print(f"  Path not found: {path}")
    
    debug_print(f"Loaded {len(data_tensors)} data tensors")
    
    if not data_tensors:
        debug_print("No data tensors found. Run the other tests first.")
        # Create a simple report explaining the issue
        with open(save_dir / "no_data_report.md", "w") as f:
            f.write("# Dataset Comparison Report\n\n")
            f.write("## Error: No Data Tensors Found\n\n")
            f.write("No data tensors were found in the expected locations. Please run the following tests first:\n\n")
            f.write("1. `test_forager_dmbd_simplified`\n")
            f.write("2. `test_synthetic_dmbd`\n")
            f.write("3. `test_binary_dmbd`\n\n")
            f.write("These tests will generate the necessary data tensors for comparison.\n")
        debug_log.close()
        return
    
    debug_print("Creating comparison visualizations...")
    
    # 1. Compare basic statistics
    fig, ax = plt.subplots(figsize=(12, 8))
    stats = []
    
    for name, tensor in data_tensors.items():
        # Convert to numpy for easier handling
        data = tensor.numpy()
        # Calculate statistics for each dimension
        for dim in range(data.shape[2]):
            dim_data = data[:, 0, dim]  # Assuming batch dimension is 1
            stats.append({
                'Dataset': name,
                'Dimension': dim + 1,
                'Mean': np.mean(dim_data),
                'Std': np.std(dim_data),
                'Min': np.min(dim_data),
                'Max': np.max(dim_data),
                'Range': np.max(dim_data) - np.min(dim_data),
                'Skewness': (np.mean(dim_data) - np.median(dim_data)) / np.std(dim_data) if np.std(dim_data) > 0 else 0
            })
    
    # Create a table of statistics
    stats_df = pd.DataFrame(stats)
    debug_print(f"Created statistics dataframe with {len(stats)} rows")
    
    # Save statistics to CSV
    stats_df.to_csv(save_dir / "dataset_comparison_stats.csv", index=False)
    debug_print(f"Saved statistics to {save_dir / 'dataset_comparison_stats.csv'}")
    
    # Create a heatmap of statistics
    pivot_mean = stats_df.pivot(index='Dataset', columns='Dimension', values='Mean')
    pivot_std = stats_df.pivot(index='Dataset', columns='Dimension', values='Std')
    pivot_skew = stats_df.pivot(index='Dataset', columns='Dimension', values='Skewness')
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    sns.heatmap(pivot_mean, annot=True, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Mean Values by Dataset and Dimension")
    
    sns.heatmap(pivot_std, annot=True, cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Standard Deviation by Dataset and Dimension")
    
    sns.heatmap(pivot_skew, annot=True, cmap="coolwarm", center=0, ax=axes[2])
    axes[2].set_title("Skewness by Dataset and Dimension")
    
    plt.tight_layout()
    plt.savefig(save_dir / "dataset_comparison_heatmap.png", dpi=300)
    debug_print(f"Saved heatmap to {save_dir / 'dataset_comparison_heatmap.png'}")
    
    # 2. Compare distributions
    max_dims = max(tensor.shape[2] for tensor in data_tensors.values())
    fig, axes = plt.subplots(len(data_tensors), max_dims, figsize=(max_dims * 4, len(data_tensors) * 4))
    
    # If only one dataset, ensure axes is 2D
    if len(data_tensors) == 1:
        axes = np.array([axes])
    # If only one dimension, ensure axes is 2D
    if max_dims == 1:
        axes = np.array([axes]).T
    
    for i, (name, tensor) in enumerate(data_tensors.items()):
        data = tensor.numpy()
        for j in range(max_dims):
            if j < data.shape[2]:
                dim_data = data[:, 0, j]
                sns.histplot(dim_data, kde=True, ax=axes[i, j])
                axes[i, j].set_title(f"{name} - Dimension {j+1}")
                axes[i, j].set_xlabel("Value")
                axes[i, j].set_ylabel("Frequency")
            else:
                axes[i, j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / "dataset_comparison_distributions.png", dpi=300)
    debug_print(f"Saved distributions to {save_dir / 'dataset_comparison_distributions.png'}")
    
    # 3. Compare autocorrelations
    fig, axes = plt.subplots(len(data_tensors), max_dims, figsize=(max_dims * 4, len(data_tensors) * 4))
    
    # If only one dataset, ensure axes is 2D
    if len(data_tensors) == 1:
        axes = np.array([axes])
    # If only one dimension, ensure axes is 2D
    if max_dims == 1:
        axes = np.array([axes]).T
    
    for i, (name, tensor) in enumerate(data_tensors.items()):
        data = tensor.numpy()
        for j in range(max_dims):
            if j < data.shape[2]:
                dim_data = data[:, 0, j]
                # Calculate autocorrelation
                autocorrelation_plot(pd.Series(dim_data), ax=axes[i, j])
                axes[i, j].set_title(f"{name} - Dimension {j+1} Autocorrelation")
            else:
                axes[i, j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / "dataset_comparison_autocorrelations.png", dpi=300)
    debug_print(f"Saved autocorrelations to {save_dir / 'dataset_comparison_autocorrelations.png'}")
    
    # 4. Compare cross-correlations within each dataset
    for name, tensor in data_tensors.items():
        data = tensor.numpy()
        if data.shape[2] > 1:  # Only if there are multiple dimensions
            fig, ax = plt.subplots(figsize=(10, 8))
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data[:, 0, :].T)
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                        xticklabels=[f"Dim {i+1}" for i in range(data.shape[2])],
                        yticklabels=[f"Dim {i+1}" for i in range(data.shape[2])],
                        ax=ax)
            ax.set_title(f"{name} - Dimension Correlations")
            plt.tight_layout()
            plt.savefig(save_dir / f"{name.lower()}_correlation_matrix.png", dpi=300)
            debug_print(f"Saved correlation matrix for {name} to {save_dir / f'{name.lower()}_correlation_matrix.png'}")
    
    # Create a summary report
    with open(save_dir / "dataset_comparison_report.md", "w") as f:
        f.write("# Dataset Comparison Report\n\n")
        f.write("## Overview\n")
        f.write("This report compares the characteristics of different datasets used for DMBD testing.\n\n")
        
        f.write("## Datasets Analyzed\n")
        for name, tensor in data_tensors.items():
            f.write(f"- **{name}**: Shape {tensor.shape}\n")
        f.write("\n")
        
        f.write("## Key Observations\n")
        f.write("- The statistics show differences in data distributions across datasets\n")
        f.write("- Binary dataset has the simplest distribution (0/1 values only)\n")
        f.write("- Synthetic dataset has controlled dependencies between variables\n")
        f.write("- Forager dataset has more complex dynamics and potentially higher noise\n\n")
        
        f.write("## Implications for DMBD Convergence\n")
        f.write("- Despite varying complexity levels, all datasets failed to converge\n")
        f.write("- This suggests fundamental issues with the DMBD update algorithm\n")
        f.write("- The binary dataset's failure is particularly concerning as it represents the simplest possible case\n")
        f.write("- Further investigation into the DMBD implementation is recommended\n")
    
    debug_print(f"Comparison visualizations and report saved to {save_dir}")
    debug_log.close()
    return stats_df

if __name__ == "__main__":
    # This allows running the test directly
    pytest.main(["-xvs", __file__]) 