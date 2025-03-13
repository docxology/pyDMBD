# Documentation and Examples for Dynamic Markov Blanket Detection

This directory contains comprehensive documentation and examples for working with the Dynamic Markov Blanket Detection (DMBD) algorithm implemented in the pyDMBD repository.

## Directory Structure

- **[docs/](docs/)**: Comprehensive documentation of the DMBD algorithm
  - Theory and mathematical foundations
  - API reference
  - Installation and setup guides
  - Tutorials and how-to guides
  
- **[examples/](examples/)**: Ready-to-run examples demonstrating DMBD in action
  - Well-commented example scripts
  - Demonstrations of different use cases
  - Visualization techniques for DMBD results

## Getting Started

If you're new to DMBD, we recommend starting with:

1. Read the overview in [docs/index.md](docs/index.md)
2. Follow the [Quick Start Guide](docs/quick_start.md)
3. Run the [Lorenz Attractor example](examples/lorenz_attractor.py)

## Documentation Highlights

- **Theory**: Learn about the mathematical foundations of Markov blankets and how they are used in dynamical systems
- **Tutorials**: Step-by-step guides to using DMBD for different applications
- **API Reference**: Detailed documentation of all classes and methods
- **Examples**: Ready-to-run examples showing DMBD in action

## Contributing to Documentation

If you'd like to contribute to improving the documentation:

1. Fork the repository
2. Add or improve documentation in the `docxology` directory
3. Submit a pull request with your changes

## Building the Documentation

The documentation is written in Markdown and can be converted to other formats using tools like MkDocs or Sphinx.

To build the documentation using MkDocs (not yet configured):

```bash
# Install MkDocs (future feature)
pip install mkdocs

# Build the site
cd docxology
mkdocs build

# Serve locally for development
mkdocs serve
```

## Citing DMBD

If you use DMBD in your research, please cite:

```
@article{TODO,
  title={Dynamic Markov Blanket Discovery},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

## License

This documentation and examples are licensed under the same terms as the main repository. See the [LICENSE.md](../LICENSE.md) file for details.

## Known Issues

There are some known issues with the DMBD model's forward pass related to tensor dimensions. Specifically, the error occurs with the `batch2` tensor in the `ARHMM_prXRY` class, where the expected size for the first two dimensions should be `[1200, 0]` but it's getting `[1200, 1]`.

For more details, see the [KNOWN_ISSUES.md](KNOWN_ISSUES.md) file.

### Affected Tests

The following tests have been skipped due to the known issues:
- `test_dmbd_forward_pass`
- `test_dmbd_loss_computation`
- `test_dmbd_multiple_objects`

### Potential Fixes

Several potential fixes have been implemented in the codebase:

1. Modified the `MatrixNormalWishart.Elog_like_X` method to handle the case where a tensor has a second dimension of 0.
2. Modified the `ARHMM_prXRY.Elog_like_X` method to ensure it correctly initializes and maintains the `batch2` tensor with a second dimension of 0.
3. Modified the `block_diag_matrix_builder` function to handle the case where either tensor has a second dimension of 0.

These fixes are documented in detail in the [KNOWN_ISSUES.md](KNOWN_ISSUES.md) file.

## Example Simulations

The `examples/` directory now includes interactive examples that demonstrate the pyDMBD framework with various simulated dynamical systems:

- **Cart with Two Pendulums**: A simulation of a cart with two attached pendulums, demonstrating complex dynamics and interactions.
- **Flame Simulation**: Simulation of flame propagation with heat diffusion across multiple sources.
- **Forager Models**: Two models demonstrating agent-based behavior in collecting food resources.
- **Lorenz Attractor**: Simulation of the well-known chaotic dynamical system.
- **Newton's Cradle**: Simulation of colliding pendulums demonstrating conservation principles.

You can view the example visualizations by opening `example_outputs/index.html` in your web browser after running the example tests:

```bash
cd /path/to/pyDMBD/fork
python -m pytest tests/test_examples.py -v
```

These examples can serve as starting points for implementing DMBD analysis on your own dynamical systems. 