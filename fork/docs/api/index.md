# API Reference

This section provides detailed documentation for the classes and methods in the DMBD package.

## Core Classes

### [DMBD](dmbd.md)

The main class for Dynamic Markov Blanket Detection. This class implements the core algorithm for discovering macroscopic objects from microscopic observations.

```python
model = DMBD(
    obs_shape,          # Shape of observables (n_observables, observable_dimension)
    role_dims,          # Number of roles for environment, boundary, internal states
    hidden_dims,        # Number of latent dimensions for environment, boundary, internal states
    control_dim=0,      # Dimension of control inputs
    regression_dim=0,   # Dimension of regression covariates
    batch_shape=(),     # Batch dimensions
    number_of_objects=1 # Number of objects to discover
)
```

### [LinearDynamicalSystems](lds.md)

The base class for linear dynamical systems, which DMBD extends.

```python
model = LinearDynamicalSystems(
    hidden_dim,         # Dimension of the latent state
    control_dim=0,      # Dimension of control inputs
    regression_dim=0,   # Dimension of regression covariates
    batch_shape=()      # Batch dimensions
)
```

### [ARHMM](arhmm.md)

Class implementing the Auto-Regressive Hidden Markov Model used for assignments in DMBD.

```python
model = ARHMM_prXRY(
    role_dim,           # Number of roles
    input_dim,          # Dimension of the input
    regression_dim=0,   # Dimension of regression covariates
    batch_shape=()      # Batch dimensions
)
```

## Distributions

### [MatrixNormalWishart](dists/matrix_normal_wishart.md)

A matrix normal Wishart distribution used for parameter posteriors.

### [NormalInverseWishart](dists/normal_inverse_wishart.md)

A normal inverse Wishart distribution used for initial state distribution.

### [MultivariateNormal](dists/multivariate_normal.md)

A multivariate normal distribution in vector format.

## Main Methods

### Update Methods

- [`update(y, u, r, iters=1, latent_iters=1, lr=1.0, verbose=False)`](dmbd.md#update): Update the model parameters and latent variables.
- [`update_latents(y, u, r, p=None, lr=1.0)`](dmbd.md#update_latents): Update the posterior over latent states.
- [`update_obs_parms(y, r, lr=1.0)`](dmbd.md#update_obs_parms): Update the parameters of the observation model.
- [`update_latent_parms(p=None, lr=1.0)`](dmbd.md#update_latent_parms): Update the parameters of the latent dynamics.

### Inference Methods

- [`log_likelihood_function(Y, R)`](dmbd.md#log_likelihood_function): Compute the log likelihood of the data.
- [`KLqprior()`](dmbd.md#klqprior): Compute the KL divergence between the posterior and prior distributions.
- [`ELBO()`](dmbd.md#elbo): Compute the evidence lower bound (ELBO) of the model.

### Assignment Methods

- [`assignment_pr()`](dmbd.md#assignment_pr): Get the assignment probabilities.
- [`assignment()`](dmbd.md#assignment): Get the hard assignments.
- [`particular_assignment_pr()`](dmbd.md#particular_assignment_pr): Get the probabilities of particular assignments.
- [`particular_assignment()`](dmbd.md#particular_assignment): Get particular hard assignments.

### Visualization Methods

- [`plot_observation()`](dmbd.md#plot_observation): Plot the observation model.
- [`plot_transition(type='obs', use_mask=False)`](dmbd.md#plot_transition): Plot the transition model.
- [`animate_results`](dmbd.md#animate_results): Class for animating the results of DMBD.

## Masking Methods

- [`one_object_mask(hidden_dims, role_dims, control_dim, obs_dim, regression_dim)`](dmbd.md#one_object_mask): Create a mask for a single object.
- [`n_object_mask(n, hidden_dims, role_dims, control_dim, obs_dim, regression_dim)`](dmbd.md#n_object_mask): Create a mask for multiple objects. 