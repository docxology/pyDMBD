# DMBD Numerical Stability Integration Guide

This guide provides instructions for integrating the numerical stability improvements into your Dynamic Markov Blanket Discovery (DMBD) implementation. These improvements address root causes of numerical instability rather than applying temporary fixes.

## Table of Contents

1. [Overview](#overview)
2. [Integration Options](#integration-options)
3. [Quick Start](#quick-start)
4. [Detailed Integration Steps](#detailed-integration-steps)
5. [Testing Your Integration](#testing-your-integration)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## Overview

The stability improvements address three key areas:

1. **Matrix Inversion Stability**: Replaces standard matrix inversion with SVD-based pseudo-inverse and regularization techniques to handle ill-conditioned matrices.

2. **Tensor Dimension Consistency**: Ensures consistent batch dimensions across tensors, with automatic broadcasting and validation.

3. **Robust Update Mechanisms**: Implements exception handling and adaptive regularization to prevent cascading failures during model updates.

## Integration Options

You have three options for integrating these improvements:

1. **Full Integration**: Copy the entire `stability_fixes.py` module and apply all fixes to your DMBD model.
2. **Selective Integration**: Apply only specific fixes that address your particular stability issues.
3. **Runtime Patching**: Use the provided functions to patch an existing DMBD instance at runtime without modifying the core code.

## Quick Start

For the fastest integration, add the `stability_fixes.py` module to your project and use:

```python
from dmbd.stability_fixes import apply_all_stability_fixes
from dmbd.dmbd import DMBD

# Create your DMBD model
model = DMBD(obs_shape=(1, 9), role_dims=[3, 3, 3], hidden_dims=[3, 3, 3], number_of_objects=1)

# Apply all stability fixes
apply_all_stability_fixes(model)

# Use the model with enhanced stability
success = model.update(y=features, u=None, r=None, iters=50, lr=0.001, verbose=True)
```

## Detailed Integration Steps

### 1. Add the Stability Fixes Module

Copy the `stability_fixes.py` file to your project's DMBD module directory:

```bash
cp /path/to/stability_fixes.py /path/to/your/project/dmbd/
```

### 2. Import the Module

In your code, import the stability fixes:

```python
try:
    from dmbd.stability_fixes import apply_all_stability_fixes
except ImportError:
    from stability_fixes import apply_all_stability_fixes
```

### 3. Apply Fixes to Your Model

Apply all fixes at once:

```python
apply_all_stability_fixes(model)
```

Or apply specific fixes:

```python
from dmbd.stability_fixes import (
    apply_matrix_inversion_fixes,
    apply_dimension_consistency_fixes,
    apply_robust_update_fixes
)

# Apply only matrix inversion fixes
apply_matrix_inversion_fixes(model)

# Apply only dimension consistency fixes
apply_dimension_consistency_fixes(model)

# Apply only robust update fixes
apply_robust_update_fixes(model)
```

### 4. Use Enhanced Update Parameters

When calling the update method, take advantage of the enhanced parameters:

```python
success = model.update(
    y=features,
    u=None,
    r=None,
    iters=50,
    lr=0.001,
    verbose=True,
    max_attempts=3,           # New parameter: maximum update attempts
    initial_reg=1e-3,         # New parameter: initial regularization value
    adaptive_reg=True,        # New parameter: enable adaptive regularization
    dimension_check=True      # New parameter: enable dimension checking
)
```

## Testing Your Integration

To verify that the stability fixes are working correctly:

1. Run the provided example script:

```bash
python dmbd_with_stability_fixes.py
```

2. Check the comparison report in the output directory to confirm improvements.

3. Run your own tests with challenging datasets that previously caused numerical issues.

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure the `stability_fixes.py` file is in your Python path.

   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

2. **AttributeError**: If you get errors about missing attributes, ensure you're using a compatible DMBD version.

3. **Performance Degradation**: If you experience slower performance, try adjusting the regularization parameters:

   ```python
   apply_all_stability_fixes(model, svd_rcond=1e-10, min_eigenvalue=1e-6)
   ```

### Logging

Enable detailed logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Customizing Matrix Inversion

You can customize the SVD-based pseudo-inverse behavior:

```python
from dmbd.stability_fixes import apply_matrix_inversion_fixes

apply_matrix_inversion_fixes(
    model,
    svd_rcond=1e-10,           # Cutoff for small singular values
    fallback_to_numpy=True,    # Use NumPy as fallback
    min_eigenvalue=1e-6        # Minimum eigenvalue for positive definiteness
)
```

### Customizing Dimension Consistency

Configure dimension consistency behavior:

```python
from dmbd.stability_fixes import apply_dimension_consistency_fixes

apply_dimension_consistency_fixes(
    model,
    auto_broadcast=True,       # Automatically broadcast dimensions
    strict_validation=False    # Allow some dimension mismatches
)
```

### Customizing Update Mechanisms

Configure the robust update behavior:

```python
from dmbd.stability_fixes import apply_robust_update_fixes

apply_robust_update_fixes(
    model,
    max_attempts=5,            # Maximum update attempts
    reg_increase_factor=2.0,   # Factor to increase regularization
    initial_reg=1e-4           # Initial regularization value
)
```

---

By following this guide, you should be able to successfully integrate the numerical stability improvements into your DMBD implementation. These improvements address the root causes of instability, resulting in more reliable and robust model performance.

For questions or issues, please open an issue on the GitHub repository. 