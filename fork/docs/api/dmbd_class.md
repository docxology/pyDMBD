# DMBD Class

The `DMBD` class is the main implementation of the Dynamic Markov Blanket Detection algorithm. It extends the `LinearDynamicalSystems` class.

## Constructor

```python
class DMBD(LinearDynamicalSystems):
    def __init__(self, 
                 obs_shape, 
                 role_dims, 
                 hidden_dims, 
                 control_dim=0, 
                 regression_dim=0, 
                 batch_shape=(),
                 number_of_objects=1, 
                 unique_obs=False):
        """
        Initialize a Dynamic Markov Blanket Detection model.
        
        Parameters:
        ----------
        obs_shape : tuple (n_obs, obs_dim)
            Shape of the observable data, where n_obs is the number of observables and
            obs_dim is the dimension of each observable.
            
        role_dims : tuple (s_roles, b_roles, z_roles)
            Controls the number of roles that each observable can play when driven by
            environment (s), boundary (b), or internal state (z).
            
        hidden_dims : tuple (s_dim, b_dim, z_dim)
            Controls the number of latent variables assigned to environment (s),
            boundary (b), and internal states (z). If you include a 4th dimension,
            it creates a global latent that is shared by all observation models.
            
        control_dim : int, default=0
            Dimension of control inputs. Set to 0 or -1 if no control is used.
            
        regression_dim : int, default=0
            Dimension of regression covariates. Set to 0 or -1 if no regression is used.
            
        batch_shape : tuple, default=()
            Batch dimensions of the model.
            
        number_of_objects : int, default=1
            Number of objects to discover. Set to 1 for a single object,
            or > 1 for multiple objects.
            
        unique_obs : bool, default=False
            Whether each observable is unique.
        """
```

## Key Attributes

### Structural Attributes

- `obs_shape`: Tuple (n_obs, obs_dim) representing the shape of observables
- `obs_dim`: Dimension of each observable
- `n_obs`: Number of observables
- `role_dims`: Tuple (s_roles, b_roles, z_roles) specifying number of roles
- `role_dim`: Total number of roles (sum of role_dims)
- `hidden_dims`: Tuple (s_dim, b_dim, z_dim) specifying number of latent variables
- `hidden_dim`: Total dimension of the latent state (sum of hidden_dims)
- `control_dim`: Dimension of control inputs
- `number_of_objects`: Number of objects to discover

### Model Components

- `obs_model`: The observation model (ARHMM)
- `px`: Posterior distribution over latent states (MultivariateNormal)
- `pAB`: Posterior distribution over dynamics parameters (MatrixNormalWishart)
- `pCD`: List of posterior distributions over observation parameters (MatrixNormalWishart)
- `A_mask`: Mask applied to the dynamics matrix A to enforce Markov blanket structure
- `B_mask`: Mask applied to the control matrix B
- `role_mask`: Mask applied to the observation matrices to enforce Markov blanket structure

## Main Methods

### Update Methods

#### update

```python
def update(self, y, u, r, iters=1, latent_iters=1, lr=1.0, verbose=False):
    """
    Update the model parameters and latent variables.
    
    Parameters:
    ----------
    y : torch.Tensor
        Observable data tensor of shape (time_steps, batch_size, n_observables, observable_dimension).
    
    u : torch.Tensor or None
        Control inputs tensor of shape (time_steps, batch_size, control_dimension).
        Set to None if no control input is used.
    
    r : torch.Tensor or None
        Regression covariates tensor of shape (time_steps, batch_size, regression_dimension).
        Set to None if no regression covariates are used.
    
    iters : int, default=1
        Number of update iterations to perform.
    
    latent_iters : int, default=1
        Number of latent update iterations per parameter update.
    
    lr : float, default=1.0
        Learning rate for parameter updates (0.0 < lr <= 1.0).
    
    verbose : bool, default=False
        Whether to print progress during updates.
    
    Returns:
    -------
    ELBO : float
        The final evidence lower bound after updates.
    """
```

#### update_latents

```python
def update_latents(self, y, u, r, p=None, lr=1.0):
    """
    Update the posterior over latent states.
    
    Parameters:
    ----------
    y : torch.Tensor
        Observable data.
    
    u : torch.Tensor or None
        Control inputs.
    
    r : torch.Tensor or None
        Regression covariates.
    
    p : torch.Tensor or None, default=None
        Assignment probabilities. If None, they are computed from the observation model.
    
    lr : float, default=1.0
        Learning rate for updates.
    
    Returns:
    -------
    log_like : float
        The log likelihood of the data under the current model.
    """
```

#### update_obs_parms

```python
def update_obs_parms(self, y, r, lr=1.0):
    """
    Update the parameters of the observation model.
    
    Parameters:
    ----------
    y : torch.Tensor
        Observable data.
    
    r : torch.Tensor or None
        Regression covariates.
    
    lr : float, default=1.0
        Learning rate for updates.
    """
```

#### update_latent_parms

```python
def update_latent_parms(self, p=None, lr=1.0):
    """
    Update the parameters of the latent dynamics.
    
    Parameters:
    ----------
    p : torch.Tensor or None, default=None
        Assignment probabilities.
    
    lr : float, default=1.0
        Learning rate for updates.
    """
```

### Assignment Methods

#### assignment_pr

```python
def assignment_pr(self):
    """
    Get the assignment probabilities.
    
    Returns:
    -------
    torch.Tensor
        Tensor of shape (time_steps, batch_size, n_observables) containing
        the probability of each observable being assigned to environment (0),
        boundary (1), or object (2).
    """
```

#### assignment

```python
def assignment(self):
    """
    Get the hard assignments based on the highest probability.
    
    Returns:
    -------
    torch.Tensor
        Tensor of shape (time_steps, batch_size, n_observables) containing
        integer assignments: environment (0), boundary (1), or object (2).
    """
```

#### particular_assignment_pr

```python
def particular_assignment_pr(self):
    """
    Get the probabilities of particular role assignments.
    
    Returns:
    -------
    torch.Tensor
        Tensor containing the probability of each observable being assigned to
        each particular role.
    """
```

#### particular_assignment

```python
def particular_assignment(self):
    """
    Get the hard assignments to particular roles based on the highest probability.
    
    Returns:
    -------
    torch.Tensor
        Tensor containing integer assignments to particular roles.
    """
```

### Inference Methods

#### log_likelihood_function

```python
def log_likelihood_function(self, Y, R):
    """
    Compute the log likelihood of the data.
    
    Parameters:
    ----------
    Y : torch.Tensor
        Observable data.
    
    R : torch.Tensor or None
        Regression covariates.
    
    Returns:
    -------
    tuple
        Tuple containing the log likelihood components.
    """
```

#### KLqprior

```python
def KLqprior(self):
    """
    Compute the KL divergence between the posterior and prior distributions.
    
    Returns:
    -------
    float
        The KL divergence.
    """
```

#### ELBO

```python
def ELBO(self):
    """
    Compute the evidence lower bound (ELBO) of the model.
    
    Returns:
    -------
    float
        The ELBO value.
    """
```

### Masking Methods

#### one_object_mask

```python
def one_object_mask(self, hidden_dims, role_dims, control_dim, obs_dim, regression_dim):
    """
    Create a mask for a single object.
    
    This defines the Markov blanket structure for a single object in the environment.
    
    Parameters:
    ----------
    hidden_dims : tuple (s_dim, b_dim, z_dim)
        Number of latent dimensions for environment, boundary, and object.
    
    role_dims : tuple (s_roles, b_roles, z_roles)
        Number of roles for environment, boundary, and object.
    
    control_dim : int
        Dimension of control inputs.
    
    obs_dim : int
        Dimension of observables.
    
    regression_dim : int
        Dimension of regression covariates.
    
    Returns:
    -------
    tuple
        Tuple of masks (A_mask, B_mask, role_mask) for dynamics and observation matrices.
    """
```

#### n_object_mask

```python
def n_object_mask(self, n, hidden_dims, role_dims, control_dim, obs_dim, regression_dim):
    """
    Create a mask for multiple objects.
    
    This defines the Markov blanket structure for multiple objects in a shared environment.
    
    Parameters:
    ----------
    n : int
        Number of objects.
    
    hidden_dims : tuple (s_dim, b_dim, z_dim)
        Number of latent dimensions for environment, boundary, and object.
    
    role_dims : tuple (s_roles, b_roles, z_roles)
        Number of roles for environment, boundary, and object.
    
    control_dim : int
        Dimension of control inputs.
    
    obs_dim : int
        Dimension of observables.
    
    regression_dim : int
        Dimension of regression covariates.
    
    Returns:
    -------
    tuple
        Tuple of masks (A_mask, B_mask, role_mask) for dynamics and observation matrices.
    """
```

### Visualization Methods

#### plot_observation

```python
def plot_observation(self):
    """
    Plot the observation model.
    
    This visualizes the emission matrices for each role.
    
    Returns:
    -------
    tuple
        Tuple of figure and axes objects.
    """
```

#### plot_transition

```python
def plot_transition(self, type='obs', use_mask=False):
    """
    Plot the transition model.
    
    Parameters:
    ----------
    type : str, default='obs'
        Type of transition to plot ('obs' or 'latent').
    
    use_mask : bool, default=False
        Whether to apply the mask to the transition matrix before plotting.
    
    Returns:
    -------
    tuple
        Tuple of figure and axes objects.
    """
```

## animate_results Class

```python
class animate_results():
    def __init__(self, assignment_type='sbz', f=r'./movie_temp.', xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), fps=20):
        """
        Initialize animation of DMBD results.
        
        Parameters:
        ----------
        assignment_type : str, default='sbz'
            Type of assignment to visualize.
        
        f : str, default='./movie_temp.'
            File prefix for saved frames.
        
        xlim : tuple, default=(-2.5, 2.5)
            X-axis limits for the plot.
        
        ylim : tuple, default=(-2.5, 2.5)
            Y-axis limits for the plot.
        
        fps : int, default=20
            Frames per second for the animation.
        """
    
    def animation_function(self, frame_number, fig_data, fig_assignments, fig_confidence):
        """
        Function to generate each frame of the animation.
        """
    
    def make_movie(self, model, data, batch_numbers):
        """
        Create a movie of the DMBD assignments over time.
        
        Parameters:
        ----------
        model : DMBD
            The trained DMBD model.
        
        data : torch.Tensor
            The observable data.
        
        batch_numbers : list or int
            Batch indices to include in the animation.
        """
``` 