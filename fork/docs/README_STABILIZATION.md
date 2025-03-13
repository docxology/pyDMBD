# DMBD Stabilization Utilities

This document explains how to use the utility functions in `dmbd_utils.py` to improve the numerical stability and reliability of the DMBD model.

## Overview

The DMBD (Dynamic Markov Blanket Discovery) model can experience several common issues:

1. **Singular matrices** during inversion operations
2. **Dimension mismatches** between tensors
3. **Numerical instability** during updates

The `dmbd_utils.py` module provides functions to address these issues, making your DMBD models more robust.

## Quick Start

```python
from dmbd.dmbd import DMBD
from dmbd.dmbd_utils import patch_model_for_stability

# Create your DMBD model
model = DMBD(
    obs_shape=(1, feature_dim),
    role_dims=[3, 3, 3],
    hidden_dims=[3, 3, 3],
    number_of_objects=1
)

# Apply stabilization patches
patch_model_for_stability(model, reg_strength=1e-3)

# Run your model (now with improved stability)
model.update(
    y=data,
    u=None,
    r=None,
    iters=10,
    lr=0.001
)

# When done, restore patches if needed
from dmbd.dmbd_utils import restore_model_patches
restore_model_patches(model)
```

## Key Utility Functions

### Matrix Regularization

```python
from dmbd.dmbd_utils import regularize_matrix, safe_matrix_inverse

# Add regularization to a potentially singular matrix
regularized_matrix = regularize_matrix(matrix, reg_strength=1e-4)

# Safely invert a matrix with automatic regularization
inverted_matrix = safe_matrix_inverse(matrix, reg_strength=1e-4)
```

### Dimension Checking

```python
from dmbd.dmbd_utils import check_tensor_dimensions, check_model_dimensions

# Check if a tensor matches expected dimensions
issues = check_tensor_dimensions(tensor, expected_shape=(20, 1, 12))
if issues:
    print("Dimension issues:", issues)

# Check all dimensions in a DMBD model
dimension_info = check_model_dimensions(model)
if dimension_info["issues"]:
    print("Model has dimension issues:", dimension_info["issues"])
```

### Model Patching

```python
from dmbd.dmbd_utils import patch_model_for_stability, apply_model_regularization

# Apply multiple stability improvements to a model
success = patch_model_for_stability(model, reg_strength=1e-3)

# Just apply regularization to matrices in the model
apply_model_regularization(model, reg_strength=1e-3)
```

## Common Issues and Solutions

### Singular Matrix Errors

Error messages like:
```
The diagonal element X is zero, the inversion could not be completed because the input matrix is singular.
```

Solution:
```python
from dmbd.dmbd_utils import patch_model_for_stability

patch_model_for_stability(model, reg_strength=1e-3)
```

### Dimension Mismatch Errors

Error messages like:
```
The expanded size of the tensor (A) must match the existing size (B) at non-singleton dimension N.
```

Solution:
```python
from dmbd.dmbd_utils import check_model_dimensions

# Check dimensions before running update
dimension_info = check_model_dimensions(model)
if dimension_info["issues"]:
    # Fix dimension issues before continuing
    print("Fix these issues:", dimension_info["issues"])
```

### NaN or Inf Values

Problem: Model produces NaN or infinite values during updates.

Solution:
```python
from dmbd.dmbd_utils import debug_tensor

# Check tensor properties
tensor_info = debug_tensor(problematic_tensor)
print(tensor_info)

# Use safe inversion
from dmbd.dmbd_utils import safe_matrix_inverse
inverted = safe_matrix_inverse(matrix, reg_strength=1e-3)
```

## Best Practices

1. **Always use regularization** with a strength of at least 1e-4 for matrix inversions
2. **Apply `patch_model_for_stability()`** before running DMBD updates
3. **Verify tensor dimensions** match with `check_model_dimensions()`
4. **Use `safe_matrix_inverse()`** for all matrix inversions in custom code
5. **Check tensors** with `debug_tensor()` when troubleshooting

## Complete Example

Here's a complete example showing how to use the stabilization utilities:

```python
import torch
import numpy as np
from dmbd.dmbd import DMBD
from dmbd.dmbd_utils import (
    patch_model_for_stability,
    restore_model_patches,
    check_model_dimensions,
    debug_tensor
)

# Create test data
time_steps = 20
feature_dim = 9
batch_size = 1
data = torch.randn(time_steps, batch_size, feature_dim)

# Create model
model = DMBD(
    obs_shape=(batch_size, feature_dim),
    role_dims=[3, 3, 3],
    hidden_dims=[3, 3, 3],
    number_of_objects=1
)

# Check model dimensions
dimension_info = check_model_dimensions(model)
if dimension_info["issues"]:
    print("Dimension issues found:")
    for issue in dimension_info["issues"]:
        print(f"  - {issue}")

# Apply stabilization
success = patch_model_for_stability(model, reg_strength=1e-3)
print(f"Applied stabilization: {success}")

# Run model
try:
    success = model.update(
        y=data,
        u=None,
        r=None,
        iters=10,
        lr=0.001,
        verbose=True
    )
    print(f"Update success: {success}")
    
except Exception as e:
    print(f"Update failed: {str(e)}")

# Restore patches
restore_model_patches(model)
```

## Testing Your Model's Stability

You can use the `dmbd_stabilized_example.py` script to test your model's stability with different configurations and see the effects of these utilities.

```bash
python fork/examples/dmbd_stabilized_example.py
```

This will run various tests and generate a report in the `stabilized_outputs` directory. 