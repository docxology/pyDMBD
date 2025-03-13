# Quick Start Guide

This guide will help you get started with Dynamic Markov Blanket Detection (DMBD) quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/bayesianempirimancer/pyDMBD.git
cd pyDMBD

# Install dependencies
pip install torch numpy matplotlib
```

## Basic Usage

Here's a minimal example to get you started with DMBD:

```python
import torch
from DynamicMarkovBlanketDiscovery import DMBD
import matplotlib.pyplot as plt

# Create or load your data
# data shape should be (time_steps, batch_size, n_observables, observable_dimension)
data = torch.randn(100, 1, 5, 4)  # Example: 100 timesteps, 1 batch, 5 observables with 4 dimensions each

# Define your model
model = DMBD(
    obs_shape=data.shape[-2:],      # Shape of observables (n_observables, observable_dimension)
    role_dims=(4, 4, 4),            # Number of roles for environment, boundary, internal states
    hidden_dims=(3, 3, 3),          # Number of latent dimensions for environment, boundary, internal states
    regression_dim=0,               # No regression covariates
    control_dim=0,                  # No control inputs
    number_of_objects=1             # Discovering a single object
)

# Train the model
for i in range(10):
    model.update(
        data,                       # Data tensor
        None,                       # No control inputs
        None,                       # No regression covariates
        iters=2,                    # Number of update iterations per call
        latent_iters=1,             # Number of latent update iterations
        lr=0.5,                     # Learning rate
        verbose=True                # Show progress
    )

# Get the assignment probabilities and classifications
assignment_probs = model.assignment_pr()  # Probabilities
assignments = model.assignment()          # Hard assignments (environment=0, boundary=1, object=2)

# Visualize results (example for 2D data)
batch_num = 0  # First batch
plt.figure()
cmap = plt.cm.get_cmap('viridis', 3)
plt.scatter(
    data[:, batch_num, :, 0].flatten(),  # x coordinates
    data[:, batch_num, :, 1].flatten(),  # y coordinates
    c=assignments[:, batch_num, :].flatten(),
    cmap=cmap,
    vmin=0,
    vmax=2
)
plt.colorbar(ticks=[0, 1, 2], label='Assignment (0=env, 1=boundary, 2=object)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('DMBD Assignments')
plt.show()
```

## Next Steps

For more detailed examples and tutorials:

1. Check out the [Tutorials](tutorials/index.md) section
2. Explore the [Examples](examples/index.md) directory for real-world applications
3. Refer to the [API Reference](api/index.md) for detailed documentation of classes and functions

## Common Parameters

- `obs_shape`: Tuple (n_observables, observable_dimension) specifying the shape of your observables
- `role_dims`: Tuple (s_roles, b_roles, z_roles) controlling the number of roles that each observable can play
- `hidden_dims`: Tuple (s_dim, b_dim, z_dim) controlling the number of latent variables
- `control_dim`: Dimension of control inputs (0 or -1 if none)
- `regression_dim`: Dimension of regression covariates (0 or -1 if none)
- `number_of_objects`: Number of objects to discover (default is 1) 