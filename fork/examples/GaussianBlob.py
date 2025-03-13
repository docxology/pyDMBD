"""
GaussianBlob example for Dynamic Markov Blanket Detection.

This module provides a simulation of a Gaussian blob moving in a circular path,
creating a clear visual example for dynamic Markov blanket detection.

The blob has a natural partitioning:
- Center (high intensity) = System (internal)
- Boundary (medium intensity) = Markov Blanket
- Outside (low intensity) = Environment (external)

Usage:
    from examples.GaussianBlob import GaussianBlobSimulation
    
    # Create the simulation
    blob_sim = GaussianBlobSimulation(grid_size=32, time_steps=200)
    
    # Run the simulation
    data, labels = blob_sim.run()
    
    # Visualize the simulation
    blob_sim.visualize(output_dir="outputs/gaussian_blob")
    
    # Use with DMBD
    from dmbd import DMBD
    dmbd_model = DMBD(obs_shape=(1, blob_sim.grid_size**2), 
                     role_dims=[...], 
                     hidden_dims=3, 
                     number_of_objects=1)
    success = dmbd_model.update(data, iterations=200, lr=0.1)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import imageio
from pathlib import Path


class GaussianBlobSimulation:
    """
    Simulation of a Gaussian blob moving in a circular path.
    
    This class provides methods to generate a dataset suitable for
    dynamic Markov blanket detection, with clear internal (system),
    boundary (blanket), and external (environment) regions.
    """
    
    def __init__(self, grid_size=32, time_steps=200, radius=None, sigma=2.0, 
                 noise_level=0.02, seed=42):
        """
        Initialize the Gaussian blob simulation.
        
        Args:
            grid_size (int): Size of the square grid (grid_size x grid_size)
            time_steps (int): Number of time steps to simulate
            radius (float): Radius of the circular path. If None, defaults to 1/3 of grid_size
            sigma (float): Standard deviation of the Gaussian blob
            noise_level (float): Amount of noise to add to the blob
            seed (int): Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.radius = radius if radius is not None else grid_size // 3
        self.sigma = sigma
        self.noise_level = noise_level
        self.seed = seed
        
        # Center of the circular path
        self.center_x = grid_size // 2
        self.center_y = grid_size // 2
        
        # Angular speed to complete one revolution in time_steps
        self.angular_speed = 2 * np.pi / time_steps
        
        # Initialize the data tensor and ground truth labels
        self.data = None
        self.ground_truth_labels = None
        self.center_positions = []
        
        # Store the last role mapping from DMBD for use in visualizations
        self._last_role_mapping = None
        
        # Set the random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def run(self):
        """
        Run the simulation and generate the dataset.
        
        Returns:
            tuple: (data tensor, ground truth labels)
                - data: Tensor of shape [time_steps, 1, grid_size*grid_size]
                - labels: Tensor of shape [time_steps, grid_size*grid_size]
        """
        # Initialize the data tensor: [time_steps, 1, grid_size*grid_size]
        data = torch.zeros((self.time_steps, 1, self.grid_size*self.grid_size), 
                          dtype=torch.float32)
        
        # Labels for ground truth
        ground_truth_labels = []
        
        # Generate the data
        for t in range(self.time_steps):
            # Calculate position on the circular path
            angle = t * self.angular_speed
            x = self.center_x + self.radius * np.cos(angle)
            y = self.center_y + self.radius * np.sin(angle)
            self.center_positions.append((x, y))
            
            # Create a grid of coordinates
            grid_x, grid_y = np.meshgrid(np.arange(self.grid_size), 
                                         np.arange(self.grid_size))
            
            # Calculate the Gaussian blob
            blob = np.exp(-((grid_x - x)**2 + (grid_y - y)**2) / (2 * self.sigma**2))
            
            # Add some noise
            noise = np.random.normal(0, self.noise_level, (self.grid_size, self.grid_size))
            blob += noise
            
            # Normalize to [0, 1]
            blob = (blob - blob.min()) / (blob.max() - blob.min())
            
            # Flatten and store in the data tensor
            data[t, 0, :] = torch.from_numpy(blob.flatten()).float()
            
            # Create ground truth labels:
            # 0: System (internal) - high intensity area (center of the blob)
            # 1: Blanket - medium intensity area (gradient boundary of the blob)
            # 2: Environment (external) - low intensity area (outside the blob)
            labels = np.zeros_like(blob)
            labels[blob > 0.7] = 0  # System
            labels[(blob > 0.3) & (blob <= 0.7)] = 1  # Blanket
            labels[blob <= 0.3] = 2  # Environment
            
            ground_truth_labels.append(labels.flatten())
        
        # Save the data and labels
        self.data = data
        # Convert list to numpy array first to prevent slow tensor creation warning
        ground_truth_labels_array = np.array(ground_truth_labels)
        self.ground_truth_labels = torch.tensor(ground_truth_labels_array, dtype=torch.long)
        
        return self.data, self.ground_truth_labels
    
    def visualize(self, output_dir=None, show_animation=False):
        """
        Visualize the Gaussian blob simulation.
        
        Args:
            output_dir (str): Directory to save visualizations
            show_animation (bool): Whether to display the animation (if in interactive mode)
            
        Returns:
            str: Path to the saved animation file
        """
        if self.data is None:
            raise ValueError("Run the simulation first using run() method.")
        
        if output_dir is None:
            output_dir = "outputs/gaussian_blob"
        
        # Create the output directory
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a subdirectory for frames
        frames_dir = output_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save a few key frames
        for t in range(0, self.time_steps, self.time_steps // 10):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Raw data (Gaussian blob)
            img_data = self.data[t, 0, :].numpy().reshape(self.grid_size, self.grid_size)
            im = axes[0].imshow(img_data, cmap='viridis')
            axes[0].set_title(f"Gaussian Blob at t={t}")
            plt.colorbar(im, ax=axes[0])
            
            # Ground truth Markov blanket structure
            role_colors = [(0.2, 0.4, 0.8), (0.9, 0.3, 0.3), (0.2, 0.7, 0.2)]
            role_cmap = LinearSegmentedColormap.from_list("roles", role_colors, N=3)
            
            label_img = self.ground_truth_labels[t].reshape(self.grid_size, self.grid_size)
            im2 = axes[1].imshow(label_img, cmap=role_cmap, vmin=0, vmax=2)
            axes[1].set_title("Markov Blanket Structure\nBlue: System, Red: Blanket, Green: Environment")
            
            plt.tight_layout()
            plt.savefig(frames_dir / f"blob_t{t:03d}.png", dpi=150)
            plt.close()
        
        # Create an animation
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        def animate(t):
            for ax in axes:
                ax.clear()
            
            # Raw data (Gaussian blob)
            img_data = self.data[t, 0, :].numpy().reshape(self.grid_size, self.grid_size)
            im = axes[0].imshow(img_data, cmap='viridis', vmin=0, vmax=1)
            axes[0].set_title(f"Gaussian Blob at t={t}")
            
            # Ground truth Markov blanket structure
            label_img = self.ground_truth_labels[t].reshape(self.grid_size, self.grid_size)
            im2 = axes[1].imshow(label_img, cmap=role_cmap, vmin=0, vmax=2)
            axes[1].set_title("Markov Blanket Structure")
            
            return [im, im2]
        
        ani = animation.FuncAnimation(fig, animate, frames=range(0, self.time_steps, 2), interval=100)
        animation_path = output_dir / "blob_animation.gif"
        ani.save(animation_path, writer='pillow', fps=10)
        plt.close()
        
        if show_animation and plt.isinteractive():
            from IPython.display import display, Image
            display(Image(str(animation_path)))
        
        # Create a 3D visualization of one frame
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Pick a representative frame
            t = self.time_steps // 2
            
            # Get the data for this frame
            img_data = self.data[t, 0, :].numpy().reshape(self.grid_size, self.grid_size)
            label_img = self.ground_truth_labels[t].reshape(self.grid_size, self.grid_size)
            
            # Create a 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create coordinate grid
            X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
            
            # Plot the surface with colors indicating roles
            # We'll use viridis colormap for visualization
            surf = ax.plot_surface(
                X, Y, img_data, 
                facecolors=plt.cm.viridis(img_data),
                rstride=1, cstride=1,
                linewidth=0, antialiased=False
            )
            
            ax.set_title(f"3D View of Gaussian Blob at t={t}")
            ax.set_zlim(0, 1.2)
            
            plt.savefig(output_dir / "blob_3d_view.png", dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
        
        return str(animation_path)
    
    def analyze_with_dmbd(self, model_results, output_dir=None):
        """
        Visualize the DMBD model results compared to ground truth.
        
        Args:
            model_results (dict): Results from the DMBD model
            output_dir (str): Directory to save visualizations
            
        Returns:
            float: Accuracy of the DMBD model compared to ground truth
        """
        if self.data is None or self.ground_truth_labels is None:
            raise ValueError("Run the simulation first using run() method.")
        
        if output_dir is None:
            output_dir = "outputs/gaussian_blob_dmbd"
        
        # Create the output directory
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the DMBD role assignments
        if 'assignments' in model_results and model_results['assignments'] is not None:
            dmbd_assignments = model_results['assignments'].cpu()
        else:
            print("Warning: Model results do not contain valid role assignments")
            # Create a default assignment tensor assigning everything to role 0
            dmbd_assignments = torch.zeros((self.time_steps, 1), dtype=torch.long)
            model_results['assignments'] = dmbd_assignments
        
        # Use the last time step for mapping (steady state)
        last_frame_gt = self.ground_truth_labels[-1].long()
        
        print(f"DMBD assignments shape: {dmbd_assignments.shape}")
        print(f"Last frame ground truth shape: {last_frame_gt.shape}")
        
        # Handle different tensor dimensions
        try:
            if len(dmbd_assignments.shape) == 3:
                # Shape: [time_steps, channels, grid_points]
                last_frame_dmbd = dmbd_assignments[-1, 0, :].long()
            elif len(dmbd_assignments.shape) == 2:
                if dmbd_assignments.shape[1] == 1:
                    # Shape: [time_steps, 1] - single value per timestep
                    # Create a tensor with the same value repeated for all grid points
                    last_frame_dmbd = torch.full_like(last_frame_gt, 
                                                     dmbd_assignments[-1, 0].item(), 
                                                     dtype=torch.long)
                else:
                    # Shape: [time_steps, grid_points]
                    last_frame_dmbd = dmbd_assignments[-1, :].long()
            elif len(dmbd_assignments.shape) == 1:
                # Shape: [time_steps] - single value per timestep
                # Create a tensor with the same value repeated for all grid points
                last_frame_dmbd = torch.full_like(last_frame_gt, 
                                                 dmbd_assignments[-1].item(), 
                                                 dtype=torch.long)
            else:
                print(f"Unexpected shape for dmbd_assignments: {dmbd_assignments.shape}")
                # Create a default tensor with all zeros
                last_frame_dmbd = torch.zeros_like(last_frame_gt, dtype=torch.long)
        except Exception as e:
            print(f"Error processing dmbd_assignments: {e}")
            # Create a default tensor with all zeros
            last_frame_dmbd = torch.zeros_like(last_frame_gt, dtype=torch.long)
        
        # Map DMBD roles to ground truth roles
        # DMBD might assign different indices, so we need to find the mapping
        role_mapping = {}
        
        # Get unique roles in last_frame_dmbd
        try:
            unique_roles = torch.unique(last_frame_dmbd).tolist()
            print(f"Unique DMBD roles in last frame: {unique_roles}")
        except Exception as e:
            print(f"Error getting unique roles: {e}")
            unique_roles = [0]  # Default to a single role
        
        # Find mapping between DMBD roles and ground truth roles
        for dmbd_role in unique_roles:
            try:
                # Find pixels with this DMBD role
                mask = (last_frame_dmbd == dmbd_role)
                
                # Check if shapes match for indexing
                if mask.shape != last_frame_gt.shape:
                    print(f"Shape mismatch: mask {mask.shape}, last_frame_gt {last_frame_gt.shape}")
                    
                    # Try reshaping based on the specific shapes
                    if mask.numel() == 1:
                        # Single value mask - apply to all points
                        expanded_mask = torch.full_like(last_frame_gt, mask.item(), dtype=torch.bool)
                        mask = expanded_mask
                    elif mask.numel() != last_frame_gt.numel():
                        if mask.numel() < last_frame_gt.numel():
                            # Create larger mask (pad with False)
                            expanded_mask = torch.zeros_like(last_frame_gt, dtype=torch.bool)
                            expanded_mask[:mask.numel()] = mask.reshape(-1)
                            mask = expanded_mask
                        else:
                            # Truncate larger mask
                            mask = mask.reshape(-1)[:last_frame_gt.numel()].reshape(last_frame_gt.shape)
                    else:
                        # Same number of elements but different shape
                        mask = mask.reshape(last_frame_gt.shape)
                
                # Skip if no pixels have this role
                if mask.sum() == 0:
                    role_mapping[dmbd_role] = dmbd_role  # Default mapping
                    continue
                
                # Count ground truth roles for these pixels
                try:
                    gt_roles = last_frame_gt[mask]
                    role_counts = {}
                    for role in range(3):  # Assuming 3 roles: internal, blanket, environment
                        role_counts[role] = (gt_roles == role).sum().item()
                    
                    # Map to the most common ground truth role
                    most_common_role = max(role_counts, key=role_counts.get)
                    role_mapping[dmbd_role] = most_common_role
                    
                    print(f"DMBD role {dmbd_role} maps to ground truth role {most_common_role} (counts: {role_counts})")
                except Exception as e:
                    print(f"Error in role counting: {e}")
                    # Assign a default role mapping based on role number
                    role_mapping[dmbd_role] = min(dmbd_role, 2)  # Clip to max role 2
            except Exception as e:
                print(f"Error processing role {dmbd_role}: {e}")
                role_mapping[dmbd_role] = min(dmbd_role, 2)  # Clip to max role 2
        
        # If role_mapping is empty, create a default mapping
        if not role_mapping:
            role_mapping = {0: 0, 1: 1, 2: 2}
            print("Using default role mapping")
        
        # Store the role mapping for later use
        self._last_role_mapping = role_mapping
        print(f"Role mapping: {role_mapping}")
        
        # Map DMBD assignments to ground truth roles
        try:
            mapped_assignments = torch.zeros_like(last_frame_dmbd)
            for dmbd_role, gt_role in role_mapping.items():
                mapped_assignments[last_frame_dmbd == dmbd_role] = gt_role
            
            # Calculate accuracy
            accuracy = (mapped_assignments == last_frame_gt).float().mean().item()
            print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error mapping assignments: {e}")
            accuracy = 0.0
        
        # Create an animation comparing ground truth and DMBD assignments
        try:
            self._create_comparison_animation(dmbd_assignments, role_mapping, accuracy, output_dir)
        except Exception as e:
            print(f"Error creating comparison animation: {e}")
        
        # Save a summary image for a few key frames
        try:
            self._create_comparison_frames(dmbd_assignments, role_mapping, accuracy, output_dir)
        except Exception as e:
            print(f"Error creating comparison frames: {e}")
        
        return accuracy
    
    def _create_comparison_animation(self, dmbd_assignments, role_mapping, accuracy, output_dir):
        """Helper method to create animation comparing ground truth and DMBD roles"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Create custom colormap for roles
        role_colors = [(0.2, 0.4, 0.8), (0.9, 0.3, 0.3), (0.2, 0.7, 0.2)]
        role_cmap = LinearSegmentedColormap.from_list("roles", role_colors, N=3)
        
        def animate_comparison(t):
            for ax in axes:
                ax.clear()
            
            # Original blob
            img_data = self.data[t, 0, :].numpy().reshape(self.grid_size, self.grid_size)
            axes[0].imshow(img_data, cmap='viridis')
            axes[0].set_title(f"Gaussian Blob (t={t})")
            
            # Ground truth roles
            axes[1].imshow(self.ground_truth_labels[t].reshape(self.grid_size, self.grid_size), 
                          cmap=role_cmap, vmin=0, vmax=2)
            axes[1].set_title("Ground Truth Roles\nBlue: Internal, Red: Blanket, Green: External")
            
            # DMBD assignments (mapped to ground truth)
            # Handle different shapes of dmbd_assignments
            try:
                if len(dmbd_assignments.shape) == 3:
                    dmbd_frame = dmbd_assignments[t, 0, :].long()
                elif len(dmbd_assignments.shape) == 2:
                    if dmbd_assignments.shape[1] == 1:
                        # Create a tensor of the same value for all grid points
                        dmbd_frame = torch.full((self.grid_size * self.grid_size,), 
                                            dmbd_assignments[t, 0].item(), 
                                            dtype=torch.long)
                    else:
                        dmbd_frame = dmbd_assignments[t, :].long()
                else:
                    print(f"Unexpected shape for dmbd_assignments in animation: {dmbd_assignments.shape}")
                    # Create a default frame with all zeros
                    dmbd_frame = torch.zeros(self.grid_size * self.grid_size, dtype=torch.long)
            except Exception as e:
                print(f"Error handling DMBD assignments tensor: {e}")
                # Handle error case
                dmbd_frame = torch.zeros(self.grid_size * self.grid_size, dtype=torch.long)
            
            # Now map the dmbd frame using the role mapping
            try:
                mapped_frame = torch.zeros_like(dmbd_frame)
                for dmbd_role, gt_role in role_mapping.items():
                    mapped_frame[dmbd_frame == dmbd_role] = gt_role
                
                axes[2].imshow(mapped_frame.reshape(self.grid_size, self.grid_size), 
                            cmap=role_cmap, vmin=0, vmax=2)
                axes[2].set_title(f"DMBD Discovered Roles\nAccuracy: {accuracy:.2f}")
            except Exception as e:
                print(f"Error mapping DMBD roles: {e}")
                # Display error message
                axes[2].text(0.5, 0.5, "Error processing DMBD data", ha='center', va='center')
                axes[2].set_title("Error")
            
            return axes
        
        try:
            ani = animation.FuncAnimation(fig, animate_comparison, 
                                        frames=range(0, self.time_steps, 2), interval=100)
            ani.save(output_dir / "dmbd_comparison.gif", writer='pillow', fps=10)
        except Exception as e:
            print(f"Error creating animation: {e}")
        finally:
            plt.close()
    
    def _create_comparison_frames(self, dmbd_assignments, role_mapping, accuracy, output_dir):
        """Helper method to create comparison frames at key time points"""
        frame_indices = [0, self.time_steps // 4, self.time_steps // 2, 
                        3 * self.time_steps // 4, self.time_steps - 1]
        
        # Create custom colormap for roles
        role_colors = [(0.2, 0.4, 0.8), (0.9, 0.3, 0.3), (0.2, 0.7, 0.2)]
        role_cmap = LinearSegmentedColormap.from_list("roles", role_colors, N=3)
        
        for t in frame_indices:
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original blob
                img_data = self.data[t, 0, :].numpy().reshape(self.grid_size, self.grid_size)
                axes[0].imshow(img_data, cmap='viridis')
                axes[0].set_title(f"Gaussian Blob (t={t})")
                
                # Ground truth roles
                axes[1].imshow(self.ground_truth_labels[t].reshape(self.grid_size, self.grid_size), 
                              cmap=role_cmap, vmin=0, vmax=2)
                axes[1].set_title("Ground Truth Roles\nBlue: Internal, Red: Blanket, Green: External")
                
                # Extract DMBD frame for this timestep
                try:
                    if len(dmbd_assignments.shape) == 3:
                        dmbd_frame = dmbd_assignments[t, 0, :].long()
                    elif len(dmbd_assignments.shape) == 2:
                        if dmbd_assignments.shape[1] == 1:
                            # Create a tensor of the same value for all grid points
                            dmbd_frame = torch.full((self.grid_size * self.grid_size,), 
                                                dmbd_assignments[t, 0].item(), 
                                                dtype=torch.long)
                        else:
                            dmbd_frame = dmbd_assignments[t, :].long()
                    elif len(dmbd_assignments.shape) == 1:
                        # Create a tensor of the same value for all grid points
                        dmbd_frame = torch.full((self.grid_size * self.grid_size,), 
                                            dmbd_assignments[t].item(), 
                                            dtype=torch.long)
                    else:
                        # Default to all zeros
                        dmbd_frame = torch.zeros(self.grid_size * self.grid_size, dtype=torch.long)
                except Exception as e:
                    print(f"Error extracting DMBD frame at t={t}: {e}")
                    # Default to all zeros
                    dmbd_frame = torch.zeros(self.grid_size * self.grid_size, dtype=torch.long)
                
                # Map DMBD roles to ground truth roles
                try:
                    mapped_frame = torch.zeros_like(dmbd_frame)
                    for dmbd_role, gt_role in role_mapping.items():
                        mapped_frame[dmbd_frame == dmbd_role] = gt_role
                    
                    axes[2].imshow(mapped_frame.reshape(self.grid_size, self.grid_size), 
                                  cmap=role_cmap, vmin=0, vmax=2)
                    axes[2].set_title(f"DMBD Discovered Roles\nAccuracy: {accuracy:.2f}")
                except Exception as e:
                    print(f"Error mapping DMBD frame at t={t}: {e}")
                    # Display error message in the plot
                    axes[2].text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                              ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title("DMBD Mapping Error")
                
                plt.tight_layout()
                plt.savefig(output_dir / f"dmbd_comparison_t{t:03d}.png", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating comparison frame at t={t}: {e}")


# Example usage
if __name__ == "__main__":
    # Create the simulation
    blob_sim = GaussianBlobSimulation(grid_size=32, time_steps=200)
    
    # Run the simulation
    data, labels = blob_sim.run()
    
    # Visualize the simulation
    animation_path = blob_sim.visualize("outputs/gaussian_blob")
    print(f"Animation saved to {animation_path}")
    
    # Try running with DMBD (commented out to avoid dependencies if just viewing the example)
    """
    from dmbd import DMBD
    
    # Set up DMBD model
    grid_size = blob_sim.grid_size
    role_dims = [grid_size*grid_size // 3, grid_size*grid_size // 3, grid_size*grid_size // 3]
    dmbd_model = DMBD(obs_shape=(1, grid_size*grid_size), 
                     role_dims=role_dims, 
                     hidden_dims=3, 
                     number_of_objects=1)
    
    # Try multiple configurations for better convergence
    configs = [
        {"lr": 0.01, "iterations": 100},
        {"lr": 0.05, "iterations": 200},
        {"lr": 0.1, "iterations": 300},
        {"lr": 0.2, "iterations": 500}
    ]
    
    for config in configs:
        lr = config["lr"]
        iterations = config["iterations"]
        print(f"Trying DMBD update with lr={lr}, iterations={iterations}")
        
        success = dmbd_model.update(data, iterations=iterations, lr=lr, device="cpu", verbose=True)
        print(f"Update success: {success}")
        
        if success:
            print("DMBD update succeeded!")
            break
    
    if success:
        # Create results dictionary
        results = {
            "assignments": dmbd_model.assignments,
            "roles": dmbd_model.roles,
            "u": dmbd_model.u
        }
        
        # Analyze the results
        accuracy = blob_sim.analyze_with_dmbd(results, "outputs/gaussian_blob_dmbd")
        print(f"DMBD accuracy: {accuracy:.4f}")
    else:
        print("DMBD failed to converge on all configurations.")
    """ 