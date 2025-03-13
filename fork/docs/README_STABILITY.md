# DMBD Numerical Stability Improvements

This repository contains comprehensive improvements to address numerical stability issues in the Dynamic Markov Blanket Discovery (DMBD) implementation. Rather than applying temporary fixes or workarounds, we've focused on identifying and resolving the root causes of instability.

## Key Components

1. **[stability_fixes.py](dmbd/stability_fixes.py)**: Core module containing all stability improvements
2. **[dmbd_with_stability_fixes.py](examples/dmbd_with_stability_fixes.py)**: Example script demonstrating the improvements
3. **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**: Detailed guide for integrating the fixes into your code
4. **[STABILITY_APPROACH.md](STABILITY_APPROACH.md)**: Technical document explaining our approach to solving stability issues

## Key Improvements

### 1. Robust Matrix Operations

- **SVD-based Pseudo-inverse**: Using Singular Value Decomposition for numerically stable matrix inversion
- **Adaptive Regularization**: Automatically adjusting regularization strength based on matrix conditioning
- **Positive Definiteness Enforcement**: Ensuring matrices are positive definite before inversion

### 2. Tensor Dimension Consistency

- **Automatic Broadcasting**: Intelligently broadcasting tensors to compatible dimensions
- **Dimension Validation**: Pre-checking tensor dimensions before operations
- **Shape Adaptation**: Reshaping tensors when safe to do so

### 3. Resilient Update Mechanisms

- **Exception Handling**: Gracefully handling failures during the update process
- **Multiple Update Attempts**: Automatically retrying with adjusted parameters
- **Incremental Regularization**: Progressively increasing regularization when needed

## Quick Start

```python
from dmbd.stability_fixes import apply_all_stability_fixes
from dmbd.dmbd import DMBD

# Create your DMBD model
model = DMBD(obs_shape=(1, 9), role_dims=[3, 3, 3], hidden_dims=[3, 3, 3], number_of_objects=1)

# Apply all stability fixes
apply_all_stability_fixes(model)

# Use the model with enhanced stability
success = model.update(
    y=features,
    u=None,
    r=None,
    iters=50,
    lr=0.001,
    verbose=True,
    max_attempts=3,           # New parameter: maximum update attempts
    initial_reg=1e-3          # New parameter: initial regularization value
)
```

## Comparison with Standard DMBD

Our example script (`dmbd_with_stability_fixes.py`) demonstrates the improvements by running DMBD with and without stability fixes on the same dataset. Key metrics include:

- **Success Rate**: Enhanced DMBD succeeds on datasets where standard DMBD fails
- **Numerical Stability**: Significantly reduced matrix inversion failures
- **Dimension Handling**: Successfully handles tensor dimension mismatches
- **Performance**: Maintains computational efficiency while improving stability

## Integration Options

You have three options for integrating these improvements:

1. **Full Integration**: Copy the entire `stability_fixes.py` module and apply all fixes to your DMBD model.
2. **Selective Integration**: Apply only specific fixes that address your particular stability issues.
3. **Runtime Patching**: Use the provided functions to patch an existing DMBD instance at runtime without modifying the core code.

See the [Integration Guide](INTEGRATION_GUIDE.md) for detailed instructions.

## Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**: Step-by-step instructions for integrating the fixes
- **[STABILITY_APPROACH.md](STABILITY_APPROACH.md)**: Technical explanation of our approach
- **[PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md)**: Template for contributing your fixes back to the main repository

## Contributing

If you've made additional improvements to DMBD stability, please consider contributing them back to the repository. Use the provided pull request template to ensure your contributions are well-documented and tested.

## License

This project is licensed under the same license as the original DMBD implementation. 