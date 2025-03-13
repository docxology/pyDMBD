# Getting Started with DMBD

This tutorial will guide you through the basics of setting up and running a Dynamic Markov Blanket Detection (DMBD) model.

## Prerequisites

Before starting, make sure you have:

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib

You can install these using pip:

```bash
pip install torch numpy matplotlib
```

## Step 1: Import Required Libraries

First, let's import the necessary libraries:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from DynamicMarkovBlanketDiscovery import DMBD
```

## Step 2: Prepare Your Data

For DMBD, your data should be in the following format:

```python
data = torch.tensor(...).reshape(time_steps, batch_size, n_observables, observable_dimension)
```

Where:
- `time_steps`: Number of time steps in your sequence
- `batch_size`: Number of independent sequences/examples
- `n_observables`: Number of observables/particles
- `observable_dimension`: Dimension of each observable (e.g., position, velocity, etc.)

For this tutorial, we'll create some synthetic data:

```python
# Create synthetic data representing 3 particles orbiting a center
time_steps = 100
batch_size = 1
n_observables = 3
obs_dim = 2  # 2D positions

data = torch.zeros((time_steps, batch_size, n_observables, obs_dim))

# Initialize positions
angles = torch.tensor([0, 2*np.pi/3, 4*np.pi/3])
radius = 1.0

# Create circular motion
for t in range(time_steps):
    current_angles = angles + t * 0.1
    x = radius * torch.cos(current_angles)
    y = radius * torch.sin(current_angles)
    data[t, 0, :, 0] = x
    data[t, 0, :, 1] = y

# Add some noise
data = data + 0.05 * torch.randn_like(data)
```

## Step 3: Initialize the DMBD Model

Now, we'll create a DMBD model:

```python
# Define parameters
role_dims = (2, 2, 2)  # 2 roles each for environment, boundary, and object
hidden_dims = (2, 2, 2)  # 2 dimensions each for environment, boundary, and object latents

# Create the model
model = DMBD(
    obs_shape=data.shape[-2:],  # (n_observables, observable_dimension)
    role_dims=role_dims,
    hidden_dims=hidden_dims,
    regression_dim=0,  # No regression covariates
    control_dim=0,     # No control inputs
    number_of_objects=1
)
```

## Step 4: Train the Model

Now we can train the model using the `update` method:

```python
# Number of training iterations
training_iters = 10

for i in range(training_iters):
    elbo = model.update(
        data,     # Observable data
        None,     # No control inputs
        None,     # No regression covariates
        iters=2,  # Number of update iterations per call
        latent_iters=1,  # Number of latent update iterations
        lr=0.5,   # Learning rate
        verbose=True  # Show progress
    )
    
    print(f"Iteration {i+1}/{training_iters}, ELBO: {elbo:.4f}")
```

## Step 5: Analyze the Results

After training, we can extract the assignments and visualize the results:

```python
# Get the assignments
assignments = model.assignment()  # 0 = environment, 1 = boundary, 2 = object

# Plot the data with assignments
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
labels = ['Environment', 'Boundary', 'Object']

# Choose a batch and time point to visualize
batch_idx = 0
time_idx = -1  # Last time step

# Create a scatter plot
for i in range(3):  # For each assignment type (env, boundary, object)
    mask = (assignments[time_idx, batch_idx, :] == i)
    if mask.sum() > 0:
        plt.scatter(
            data[time_idx, batch_idx, mask, 0],
            data[time_idx, batch_idx, mask, 1],
            color=colors[i],
            label=labels[i],
            s=100
        )

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('DMBD Assignments')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Step 6: Visualize Assignments Over Time

To see how the assignments evolve over time:

```python
# Create an animation
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True)

batch_idx = 0
scatter_objs = []

for i in range(3):  # For each assignment type
    scatter = ax.scatter([], [], color=colors[i], label=labels[i], s=100)
    scatter_objs.append(scatter)

ax.legend()

def update(frame):
    for i in range(3):  # For each assignment type
        mask = (assignments[frame, batch_idx, :] == i)
        if mask.sum() > 0:
            scatter_objs[i].set_offsets(
                np.column_stack([
                    data[frame, batch_idx, mask, 0].numpy(),
                    data[frame, batch_idx, mask, 1].numpy()
                ])
            )
        else:
            scatter_objs[i].set_offsets(np.empty((0, 2)))
    
    ax.set_title(f'Frame {frame}')
    return scatter_objs

anim = FuncAnimation(fig, update, frames=time_steps, interval=100, blit=True)
plt.close()  # Prevent duplicate display in notebooks

# Save animation (optional)
# anim.save('dmbd_animation.gif', writer='pillow', fps=10)

# Display in notebook
from IPython.display import HTML
HTML(anim.to_jshtml())
```

## Next Steps

Now that you've successfully run your first DMBD model, you might want to:

1. Try adjusting the model parameters (role_dims, hidden_dims) to see how they affect the results
2. Apply the model to your own data
3. Check out the [Data Preparation](data_preparation.md) tutorial for more details on preparing data
4. Learn about [DMBD result interpretation](understanding_results.md)

## Common Issues and Solutions

### Poor Convergence

If your model doesn't converge well:
- Try increasing the number of iterations
- Reduce the learning rate (e.g., try `lr=0.1`)
- Make sure your data is properly normalized

### Memory Issues

If you encounter memory issues:
- Reduce batch size
- Reduce the number of roles or hidden dimensions
- Try processing shorter sequences

### Wrong Assignments

If assignments don't match your expectations:
- Try different initializations of the model
- Adjust the number of roles and hidden dimensions
- Make sure your data has clear patterns that match the Markov blanket structure 