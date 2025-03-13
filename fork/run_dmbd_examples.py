#!/usr/bin/env python3
"""
Run Dynamic Markov Blanket Discovery (DMBD) examples and save visualizations to output folders.

This script implements the test cases from test_dmbd.py but ensures all visualizations
are saved to disk without displaying them interactively.
"""

import os
import time
import torch
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend to prevent popups
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
from pathlib import Path
import sys
from dmbd import DMBD

# Create output directory for results
output_dir = Path(__file__).parent / "dmbd_outputs"
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for each example
forager_dir = output_dir / "forager"
flame_dir = output_dir / "flame"
lorenz_dir = output_dir / "lorenz"
newtons_cradle_dir = output_dir / "newtons_cradle"
cradle_dir = output_dir / "cradle"
calcium_dir = output_dir / "calcium"
flock_dir = output_dir / "flock"
life_dir = output_dir / "life"

# Create all directories
os.makedirs(forager_dir, exist_ok=True)
os.makedirs(flame_dir, exist_ok=True)
os.makedirs(lorenz_dir, exist_ok=True)
os.makedirs(newtons_cradle_dir, exist_ok=True)
os.makedirs(cradle_dir, exist_ok=True)
os.makedirs(calcium_dir, exist_ok=True)
os.makedirs(flock_dir, exist_ok=True)
os.makedirs(life_dir, exist_ok=True)

# Set matplotlib to non-interactive mode
plt.ioff()

def save_visualization(fig, output_dir, filename):
    """Save figure to the specified output directory."""
    path = Path(output_dir) / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved visualization to {path}")

def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def run_forager_example():
    """Run the forager simulation example."""
    print("Running forager simulation...")
    
    try:
        # Add the simulations directory to the Python path
        sys.path.append(str(Path(__file__).parent.parent / "simulations"))
        from Forager import Forager
        
        # Initialize and run simulation
        sim = Forager()
        batch_num = 100
        data, flim = sim.simulate_batches(batch_num)
        data = data/50
        
        data = data[:,::2]
        
        # Compute velocity data
        v_data = data.diff(n=1, dim=0)
        v_data[...,1:,:] = torch.zeros(v_data[...,1:,:].shape)
        v_data = v_data/v_data[:,:,0,:].std()
        data = data[1:]
        
        # Combine position and velocity
        data = torch.cat((data, v_data), -1)
        data = data[::10]
        
        # Create and train DMBD model
        model = DMBD(
            obs_shape=data.shape[-2:],
            role_dims=[4, 1, 1],
            hidden_dims=[4, 2, 0],
            batch_shape=(),
            regression_dim=0,
            control_dim=0,
            number_of_objects=10
        )
        
        # Train the model
        model.update(data, None, None, iters=20, latent_iters=1, lr=0.5)
        
        # Extract model outputs for visualization
        sbz = model.px.mean()
        B = model.obs_model.obs_dist.mean()
        if model.regression_dim == 0:
            roles = B @ sbz
        else:
            roles = B[..., :-1] @ sbz + B[..., -1:]
        sbz = sbz.squeeze()
        roles = roles.squeeze()
        
        # Get role indices
        r1 = model.role_dims[0]
        r2 = r1 + model.role_dims[1]
        r3 = r2 + model.role_dims[2]
        h1 = model.hidden_dims[0]
        h2 = h1 + model.hidden_dims[1]
        h3 = h2 + model.hidden_dims[2]
        
        # Visualization: Roles scatter plot
        batch_num = 0
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(roles[:, batch_num, list(range(0, r1)), 0], roles[:, batch_num, list(range(0, r1)), 1], color='r', alpha=0.25)
        plt.scatter(roles[:, batch_num, list(range(r2, r3)), 0], roles[:, batch_num, list(range(r2, r3)), 1], color='b', alpha=0.25)
        plt.scatter(roles[:, batch_num, list(range(r1, r2)), 0], roles[:, batch_num, list(range(r1, r2)), 1], color='g', alpha=0.25)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title("Forager Roles")
        save_visualization(fig, forager_dir, "forager_roles.png")
        
        # Assignment probabilities
        p = model.assignment_pr()
        p = p.sum(-2)
        
        # Get role components
        s = sbz[:, :, 0:h1]
        s = s - s.mean(0).mean(0)
        b = sbz[:, :, h1:h2]
        b = b - b.mean(0).mean(0)
        z = sbz[:, :, h2:h3]
        z = z - z.mean(0).mean(0)
        
        # Compute covariance matrices
        cs = (s.unsqueeze(-1) * s.unsqueeze(-2)).mean(0).mean(0)
        cb = (b.unsqueeze(-1) * b.unsqueeze(-2)).mean(0).mean(0)
        cz = (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(0).mean(0)
        
        # Get principal components
        if cs.shape[0] > 0:
            d, v = torch.linalg.eigh(cs)
            ss = v.transpose(-2, -1) @ s.unsqueeze(-1)
        else:
            ss = torch.zeros(s.shape[0], s.shape[1], 0)
            
        if cb.shape[0] > 0:
            d, v = torch.linalg.eigh(cb)
            bb = v.transpose(-2, -1) @ b.unsqueeze(-1)
        else:
            bb = torch.zeros(b.shape[0], b.shape[1], 0)
            
        if cz.shape[0] > 0:
            d, v = torch.linalg.eigh(cz)
            zz = v.transpose(-2, -1) @ z.unsqueeze(-1)
        else:
            zz = torch.zeros(z.shape[0], z.shape[1], 0)
        
        # Extract top 2 PCs
        if ss.shape[-1] >= 2:
            ss = ss.squeeze(-1)[..., -2:]
        if bb.shape[-1] >= 2:
            bb = bb.squeeze(-1)[..., -2:]
        if zz.shape[-1] >= 2:
            zz = zz.squeeze(-1)[..., -2:]
        
        # Normalize
        if ss.numel() > 0 and ss.std() > 0:
            ss = ss / ss.std()
        if bb.numel() > 0 and bb.std() > 0:
            bb = bb / bb.std()
        if zz.numel() > 0 and zz.std() > 0:
            zz = zz / zz.std()
        
        # Visualization: PC scores
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        if ss.numel() > 0:
            axs[0].plot(ss[:, batch_num, -1:].numpy(), 'r', label='s')
        if bb.numel() > 0:
            axs[0].plot(bb[:, batch_num, -1:].numpy(), 'g', label='b')
        if zz.numel() > 0:
            axs[0].plot(zz[:, batch_num, -1:].numpy(), 'b', label='z')
        
        axs[0].set_title('Top PC Score')
        axs[0].legend()
        
        if p.shape[-1] > 0:
            axs[1].plot(p[:, batch_num, 0].numpy(), 'r')
        if p.shape[-1] > 1:
            axs[1].plot(p[:, batch_num, 1].numpy(), 'g')
        if p.shape[-1] > 2:
            axs[1].plot(p[:, batch_num, 2].numpy(), 'b')
        
        axs[1].set_title('Number of Assigned Objects')
        axs[1].set_xlabel('Time')
        save_visualization(fig, forager_dir, "forager_pc_scores.png")
        
        # Save the model
        torch.save(model.state_dict(), forager_dir / "forager_model.pt")
        
        # Save the data
        torch.save(data, forager_dir / "forager_data.pt")
        
        print("Forager simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Forager simulation failed: {e}")
        return False

def run_lorenz_example():
    """Run the Lorenz attractor simulation example."""
    print("Running Lorenz attractor simulation...")
    
    try:
        # Add the simulations directory to the Python path
        sys.path.append(str(Path(__file__).parent.parent / "simulations"))
        from Lorenz import Lorenz
        
        # Create visualization colors
        cmap = ListedColormap(['red', 'green', 'blue'])
        vmin = 0  # Minimum color scale
        vmax = 2  # Maximum color scale
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Initialize and run simulation
        sim = Lorenz()
        data = sim.simulate(100)
        
        # Reshape data
        data = torch.cat((data[..., 0, :], data[..., 1, :], data[..., 2, :]), dim=-1).unsqueeze(-2)
        data = data - data.mean((0, 1, 2), True)
        
        # Create and train DMBD model
        model = DMBD(
            obs_shape=data.shape[-2:],
            role_dims=(4, 4, 4),
            hidden_dims=(4, 4, 4),
            batch_shape=(),
            regression_dim=-1,
            control_dim=0,
            number_of_objects=1
        )
        model.obs_model.ptemp = 6.0
        
        # Reference points for visualization
        loc1 = torch.tensor((-0.5, -0.6, 1.6))
        loc2 = torch.tensor((0.5, 0.6, 1.6))
        
        # Training loop
        iters = 10
        for i in range(iters):
            model.update(data, None, None, iters=2, latent_iters=1, lr=0.5)
            
            # Get model outputs for visualization
            sbz = model.px.mean().squeeze()
            r1 = model.role_dims[0]
            r2 = r1 + model.role_dims[1]
            r3 = r2 + model.role_dims[2]
            h1 = model.hidden_dims[0]
            h2 = h1 + model.hidden_dims[1]
            h3 = h2 + model.hidden_dims[2]
            
            # Get assignments
            p = model.assignment_pr()
            a = model.assignment()
            batch_num = 0
            
            # Visualization: 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                data[:, batch_num, 0, 0].numpy(),
                data[:, batch_num, 0, 2].numpy(),
                data[:, batch_num, 0, 4].numpy(),
                c=a[:, batch_num, 0].numpy(),
                cmap=cmap, norm=norm
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Lorenz Attractor (Iteration {i+1}/{iters})')
            save_visualization(fig, lorenz_dir, f"lorenz3d_iter{i+1}.png")
            
            # Visualization: 2D projections
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(
                data[:, batch_num, 0, 0].numpy(),
                data[:, batch_num, 0, 4].numpy(),
                c=a[:, batch_num, 0].numpy(),
                cmap=cmap, norm=norm
            )
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.title(f'Lorenz Attractor X-Z Projection (Iteration {i+1}/{iters})')
            save_visualization(fig, lorenz_dir, f"lorenz2d_iter{i+1}.png")
            
            # Compute distances from reference points
            d1 = (data[..., 0::2] - loc1).pow(2).sum(-1).sqrt()
            d2 = (data[..., 0::2] - loc2).pow(2).sum(-1).sqrt()
            
            # Visualization: Distance plot
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(
                d1[:, batch_num].numpy(),
                d2[:, batch_num].numpy(), 
                c=a[:, batch_num, 0].numpy(),
                cmap=cmap, norm=norm
            )
            plt.xlabel('Distance from point 1')
            plt.ylabel('Distance from point 2')
            plt.title(f'Lorenz Distances (Iteration {i+1}/{iters})')
            save_visualization(fig, lorenz_dir, f"lorenz_distances_iter{i+1}.png")
        
        # Final analysis
        p = p.sum(-2)
        
        # Get role components
        s = sbz[:, :, 0:h1]
        s = s - s.mean(0).mean(0)
        b = sbz[:, :, h1:h2]
        b = b - b.mean(0).mean(0)
        z = sbz[:, :, h2:h3]
        z = z - z.mean(0).mean(0)
        
        # Compute covariance matrices
        cs = (s.unsqueeze(-1) * s.unsqueeze(-2)).mean(0).mean(0)
        cb = (b.unsqueeze(-1) * b.unsqueeze(-2)).mean(0).mean(0)
        cz = (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(0).mean(0)
        
        # Get principal components
        d, v = torch.linalg.eigh(cs)
        ss = v.transpose(-2, -1) @ s.unsqueeze(-1)
        d, v = torch.linalg.eigh(cb)
        bb = v.transpose(-2, -1) @ b.unsqueeze(-1)
        d, v = torch.linalg.eigh(cz)
        zz = v.transpose(-2, -1) @ z.unsqueeze(-1)
        
        # Extract top 2 PCs
        ss = ss.squeeze(-1)[..., -2:]
        bb = bb.squeeze(-1)[..., -2:]
        zz = zz.squeeze(-1)[..., -2:]
        
        # Normalize
        ss = ss / ss.std()
        bb = bb / bb.std()
        zz = zz / zz.std()
        
        # Visualization: PC scores
        batch_num = 0
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        axs[0].plot(zz[:, batch_num, -1:].numpy(), 'r', label='s')
        axs[0].plot(bb[:, batch_num, -1:].numpy(), 'g', label='b')
        axs[0].plot(ss[:, batch_num, -1:].numpy(), 'b', label='z')
        axs[0].set_title('Top PC Score')
        axs[0].legend()
        
        axs[1].plot(p[:, batch_num, 2].numpy(), 'r')
        axs[1].plot(p[:, batch_num, 1].numpy(), 'g')
        axs[1].plot(p[:, batch_num, 0].numpy(), 'b')
        axs[1].set_title('Number of Assigned Nodes')
        axs[1].set_xlabel('Time')
        save_visualization(fig, lorenz_dir, "lorenz_pc_scores.png")
        
        # Save the model
        torch.save(model.state_dict(), lorenz_dir / "lorenz_model.pt")
        
        # Save the data
        torch.save(data, lorenz_dir / "lorenz_data.pt")
        
        print("Lorenz attractor simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Lorenz attractor simulation failed: {e}")
        return False

def run_flame_example():
    """Run the flame simulation example."""
    print("Running flame simulation...")
    
    try:
        # Load data
        data = torch.load('./data/flame_data.pt')
        data = data + torch.randn(data.shape) * 0.0
        data = data[:503]
        data = (data[:-2:3] + data[1:-1:3] + data[2::3]) / 3
        
        # Compute velocity data
        v_data = data.diff(n=1, dim=0)
        v_data = v_data / v_data.std((0, 1, 2), keepdim=True)
        data = torch.cat((data[1:], v_data), dim=-1)
        data = data + torch.randn(data.shape) * 0.1
        
        # Filter data
        idx = data[-1, :, 100, 0] > 0.5
        data = data[:, idx, :]
        data = data[..., 0:150, :]
        
        # Create and train DMBD model
        model = DMBD(
            obs_shape=data.shape[-2:],
            role_dims=(2, 2, 2),
            hidden_dims=(4, 4, 4),
            batch_shape=(),
            regression_dim=-1,
            control_dim=0,
            number_of_objects=1
        )
        
        # Create visualization colors
        from matplotlib.colors import ListedColormap, Normalize
        cmap = ListedColormap(['red', 'green', 'blue'])
        vmin = 0  # Minimum color scale
        vmax = 2  # Maximum color scale
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Training loop
        for i in range(10):
            model.update(data, None, None, iters=2, latent_iters=1, lr=0.5)
            
            # Get model outputs for visualization
            sbz = model.px.mean().squeeze()
            r1 = model.role_dims[0]
            r2 = r1 + model.role_dims[1]
            r3 = r2 + model.role_dims[2]
            h1 = model.hidden_dims[0]
            h2 = h1 + model.hidden_dims[1]
            h3 = h2 + model.hidden_dims[2]
            
            # Get assignments
            p = model.assignment_pr()
            a = 2 - model.assignment()
            
            # Visualization: Assignment plot
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(a[:, 0, :].transpose(-2, -1).numpy(), cmap=cmap, norm=norm, origin='lower')
            plt.xlabel('Time')
            plt.ylabel('Location')
            plt.title(f'Flame Assignments (Iteration {i+1}/10)')
            save_visualization(fig, flame_dir, f"flame_assignments_iter{i+1}.png")
            
            # Sum probabilities
            p = p.sum(-2)
            
            # Get role components
            s = sbz[:, :, 0:h1]
            s = s - s.mean(0).mean(0)
            b = sbz[:, :, h1:h2]
            b = b - b.mean(0).mean(0)
            z = sbz[:, :, h2:h3]
            z = z - z.mean(0).mean(0)
            
            # Compute covariance matrices
            cs = (s.unsqueeze(-1) * s.unsqueeze(-2)).mean(0).mean(0)
            cb = (b.unsqueeze(-1) * b.unsqueeze(-2)).mean(0).mean(0)
            cz = (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(0).mean(0)
            
            # Get principal components
            d, v = torch.linalg.eigh(cs)
            ss = v.transpose(-2, -1) @ s.unsqueeze(-1)
            d, v = torch.linalg.eigh(cb)
            bb = v.transpose(-2, -1) @ b.unsqueeze(-1)
            d, v = torch.linalg.eigh(cz)
            zz = v.transpose(-2, -1) @ z.unsqueeze(-1)
            
            # Extract top 2 PCs
            ss = ss.squeeze(-1)[..., -2:]
            bb = bb.squeeze(-1)[..., -2:]
            zz = zz.squeeze(-1)[..., -2:]
            
            # Normalize
            ss = ss / ss.std()
            bb = bb / bb.std()
            zz = zz / zz.std()
            
            # Visualization: PC scores
            batch_num = 0
            fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            axs[0].plot(zz[:, batch_num, -1:].numpy(), 'r', label='s')
            axs[0].plot(bb[:, batch_num, -1:].numpy(), 'g', label='b')
            axs[0].plot(ss[:, batch_num, -1:].numpy(), 'b', label='z')
            axs[0].set_title('Top PC Score')
            axs[0].legend()
            
            axs[1].plot(p[:, batch_num, 2].numpy(), 'r')
            axs[1].plot(p[:, batch_num, 1].numpy(), 'g')
            axs[1].plot(p[:, batch_num, 0].numpy(), 'b')
            axs[1].set_title('Number of Assigned Nodes')
            axs[1].set_xlabel('Time')
            save_visualization(fig, flame_dir, f"flame_pc_scores_iter{i+1}.png")
        
        # Save the model
        torch.save(model.state_dict(), flame_dir / "flame_model.pt")
        
        # Save the data
        torch.save(data, flame_dir / "flame_data.pt")
        
        print("Flame simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Flame simulation failed: {e}")
        return False

def smoothe(data, n):
    """Smooth the data by averaging n consecutive frames."""
    temp = data[0:-n]
    for i in range(1, n):
        temp = temp + data[i:-(n-i)]
    return temp[::n] / n

def run_flock_example():
    """Run the flocking data example."""
    print("Running flocking data analysis...")
    
    try:
        # Load data
        with np.load("data/couzin2zone_sim_hist_key1_100runs.npz") as data:
            r = data["r"]
            v = data["v"]
        
        r = torch.tensor(r).float()
        r = r / r.std()
        v = torch.tensor(v).float()
        v = v / v.std()
        
        data = torch.cat((r, v), dim=-1)
        data = data.transpose(0, 1)
        
        # Smooth the data
        data = 2 * smoothe(data, 20)
        data = data[:80]
        
        print("Preprocessing complete")
        
        # Create and train DMBD model
        model = DMBD(
            obs_shape=data.shape[-2:],
            role_dims=(2, 2, 2),
            hidden_dims=(4, 2, 2),
            regression_dim=-1,
            control_dim=0,
            number_of_objects=5,
            unique_obs=False
        )
        
        # Training loop
        iters = 40
        for i in range(iters):
            model.update(
                data[:, torch.randint(0, 50, (10,))],
                None, None,
                iters=2,
                latent_iters=4,
                lr=0.05,
                verbose=True
            )
            
            # Save progress checkpoint
            if (i + 1) % 10 == 0:
                # Save the model
                torch.save(model.state_dict(), flock_dir / f"flock_model_iter{i+1}.pt")
                print(f"Saved checkpoint at iteration {i+1}/{iters}")
        
        # Final update
        model.update(data[:, 0:4], None, None, iters=1, latent_iters=8, lr=0.0, verbose=True)
        
        # Get model outputs for visualization
        sbz = model.px.mean()
        B = model.obs_model.obs_dist.mean()
        if model.regression_dim == 1:
            roles = B[..., :-1] @ sbz + B[..., -1:]
        else:
            roles = B @ sbz
        sbz = sbz.squeeze(-3).squeeze(-1)
        roles = roles.squeeze(-1)[..., 0:2]
        
        batch_num = 0
        temp1 = data[:, batch_num, :, 0]
        temp2 = data[:, batch_num, :, 1]
        rtemp1 = roles[:, batch_num, :, 0]
        rtemp2 = roles[:, batch_num, :, 1]
        
        # Visualization: Environment and roles
        idx = (model.assignment()[:, batch_num, :] == 0)
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.5)
        
        ev_dim = model.role_dims[0]
        ob_dim = np.sum(model.role_dims[1:])
        
        for i in range(ev_dim):
            idx = (model.obs_model.assignment()[:, batch_num, :] == i)
            plt.scatter(rtemp1[:, i], rtemp2[:, i])
        
        plt.title('Environment + Roles')
        save_visualization(fig, flock_dir, "flock_env_roles.png")
        
        # Visualization: Objects
        ctemp = model.role_dims[1] * ('b',) + model.role_dims[2] * ('r',)
        
        for j in range(model.number_of_objects):
            # Object visualization
            fig = plt.figure(figsize=(10, 8))
            idx = (model.assignment()[:, batch_num, :] == 0)
            plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.2)
            
            for i in range(1 + 2 * j, 1 + 2 * (j + 1)):
                idx = (model.assignment()[:, batch_num, :] == i)
                plt.scatter(temp1[idx], temp2[idx])
            
            plt.title(f'Object {j + 1} (yellow is environment)')
            save_visualization(fig, flock_dir, f"flock_object_{j+1}.png")
            
            # Object roles visualization
            fig = plt.figure(figsize=(10, 8))
            idx = (model.assignment()[:, batch_num, :] == 0)
            plt.scatter(temp1[idx], temp2[idx], color='y', alpha=0.2)
            
            k = 0
            for i in range(ev_dim + ob_dim * j, ev_dim + ob_dim * (j + 1)):
                idx = (model.obs_model.assignment()[:, batch_num, :] == i)
                plt.scatter(rtemp1[:, i], rtemp2[:, i], color=ctemp[k])
                k = k + 1
            
            plt.title(f'Object {j + 1} roles')
            save_visualization(fig, flock_dir, f"flock_object_{j+1}_roles.png")
        
        # Save the model
        torch.save(model.state_dict(), flock_dir / "flock_model.pt")
        
        # Save the data
        torch.save(data, flock_dir / "flock_data.pt")
        
        print("Flocking data analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"Flocking data analysis failed: {e}")
        return False

def main():
    """Run all DMBD examples."""
    start_time = time.time()
    
    set_random_seed(42)
    
    # Run examples
    print("\n" + "="*50)
    print("Running DMBD Examples")
    print("="*50 + "\n")
    
    examples = [
        ("Forager", run_forager_example),
        ("Lorenz Attractor", run_lorenz_example),
        ("Flame", run_flame_example),
        ("Flocking", run_flock_example),
    ]
    
    results = {}
    
    for name, func in examples:
        print("\n" + "-"*50)
        print(f"Running {name} Example")
        print("-"*50)
        
        success = func()
        results[name] = "SUCCESS" if success else "FAILED"
    
    # Print summary
    print("\n" + "="*50)
    print("DMBD Examples Summary")
    print("="*50)
    
    for name, result in results.items():
        print(f"{name}: {result}")
    
    run_time = time.time() - start_time
    print(f"\nTotal Run Time: {run_time:.2f} seconds")
    
    # Write summary to file
    with open(output_dir / "summary.txt", "w") as f:
        f.write("DMBD Examples Summary\n")
        f.write("="*25 + "\n\n")
        
        for name, result in results.items():
            f.write(f"{name}: {result}\n")
        
        f.write(f"\nTotal Run Time: {run_time:.2f} seconds\n")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 