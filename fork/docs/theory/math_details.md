# Mathematical Details of Dynamic Markov Blanket Detection

This document provides a comprehensive mathematical description of the Dynamic Markov Blanket Detection (DMBD) algorithm, including the model structure, inference procedure, and theoretical foundations.

## Probabilistic Model

DMBD is based on a probabilistic model that combines a linear dynamical system with hidden Markov models for observable assignments.

### Latent Dynamics

The latent state $x_t$ is partitioned into three components:
- Environment variables: $s_t$
- Boundary variables: $b_t$
- Object variables: $z_t$

So $x_t = [s_t, b_t, z_t]$.

The latent dynamics follows a linear state-space model:

$$x_{t+1} = Ax_t + Bu_t + w_t,\quad w_t \sim \mathcal{N}(0, \Sigma_w)$$

where:
- $A$ is the state transition matrix
- $B$ is the control input matrix
- $u_t$ is the control input at time $t$
- $w_t$ is Gaussian process noise with covariance $\Sigma_w$

The key aspect of DMBD is the constraint on the transition matrix $A$ to enforce the Markov blanket structure:

$$A = \begin{pmatrix}
A_{ss} & A_{sb} & 0 \\
A_{bs} & A_{bb} & A_{bz} \\
0 & A_{zb} & A_{zz}
\end{pmatrix}$$

This structure enforces conditional independence between environment and object variables, with boundary variables mediating their interaction. Specifically:

1. Environment variables can only influence object variables through boundary variables
2. Object variables can only influence environment variables through boundary variables
3. Environment and object variables can directly affect themselves and boundary variables

### Observation Model

The observations are denoted by $y_t = [y_{t,1}, y_{t,2}, \ldots, y_{t,N}]$, where $N$ is the number of observables.

Each observable $y_{t,i}$ is generated based on its assignment to one of the roles. The assignment variable $\lambda_{t,i}$ determines which role is active for observable $i$ at time $t$.

The observation model is:

$$y_{t,i} = C_{\lambda_{t,i}}x_t + D_{\lambda_{t,i}}r_t + v_{\lambda_{t,i}},\quad v_{\lambda_{t,i}} \sim \mathcal{N}(0, \Sigma_{\lambda_{t,i}})$$

where:
- $C_{\lambda_{t,i}}$ is the emission matrix for role $\lambda_{t,i}$
- $D_{\lambda_{t,i}}$ is the regression matrix for role $\lambda_{t,i}$
- $r_t$ is a regression covariate
- $v_{\lambda_{t,i}}$ is Gaussian observation noise with covariance $\Sigma_{\lambda_{t,i}}$

The emission matrices $C_{\lambda}$ are also constrained based on the role type:
- Environment roles ($\lambda \in$ environment roles): $C_{\lambda}$ has zeros in columns corresponding to $b_t$ and $z_t$
- Boundary roles ($\lambda \in$ boundary roles): $C_{\lambda}$ has zeros in columns corresponding to $z_t$
- Object roles ($\lambda \in$ object roles): $C_{\lambda}$ has zeros in columns corresponding to $s_t$

This ensures that environment observables are only influenced by environment latents, boundary observables by environment and boundary latents, and object observables by boundary and object latents.

### Assignment Dynamics

The assignment variables $\lambda_{t,i}$ evolve according to a hidden Markov model with transition matrix $T$:

$$p(\lambda_{t+1,i} | \lambda_{t,i}) = T_{\lambda_{t,i}, \lambda_{t+1,i}}$$

The transition matrix $T$ is constrained to have Markov blanket structure:

$$T = \begin{pmatrix}
T_{ss} & T_{sb} & 0 \\
T_{bs} & T_{bb} & T_{bz} \\
0 & T_{zb} & T_{zz}
\end{pmatrix}$$

where:
- $T_{ss}$ is the transition probability matrix between environment roles
- $T_{sb}$ is the transition probability matrix from environment to boundary roles
- $T_{bs}$ is the transition probability matrix from boundary to environment roles
- $T_{bb}$ is the transition probability matrix between boundary roles
- $T_{bz}$ is the transition probability matrix from boundary to object roles
- $T_{zb}$ is the transition probability matrix from object to boundary roles
- $T_{zz}$ is the transition probability matrix between object roles

The zeros in $T$ enforce that environment roles can't directly transition to object roles and vice versa, preserving the Markov blanket structure over time.

## Complete Probabilistic Model

The complete joint distribution of the model is:

$$p(y_{1:T}, x_{1:T}, \lambda_{1:T} | u_{1:T}, r_{1:T}) = p(x_1) \prod_{t=1}^{T-1} p(x_{t+1} | x_t, u_t) \prod_{t=1}^T \prod_{i=1}^N p(y_{t,i} | x_t, \lambda_{t,i}, r_t) p(\lambda_{t,i} | \lambda_{t-1,i})$$

where:
- $p(x_1) = \mathcal{N}(x_1; \mu_0, \Sigma_0)$ is the initial state distribution
- $p(x_{t+1} | x_t, u_t) = \mathcal{N}(x_{t+1}; Ax_t + Bu_t, \Sigma_w)$ is the state transition probability
- $p(y_{t,i} | x_t, \lambda_{t,i}, r_t) = \mathcal{N}(y_{t,i}; C_{\lambda_{t,i}}x_t + D_{\lambda_{t,i}}r_t, \Sigma_{\lambda_{t,i}})$ is the observation likelihood
- $p(\lambda_{t,i} | \lambda_{t-1,i}) = T_{\lambda_{t-1,i}, \lambda_{t,i}}$ is the role transition probability

## Variational Inference

Exact inference in this model is intractable, so DMBD uses variational inference to approximate the posterior distribution.

### Variational Posterior

The variational posterior is factorized as:

$$q(x_{1:T}, \lambda_{1:T}) = q(x_{1:T})q(\lambda_{1:T}) = q(x_{1:T})\prod_{i=1}^N q(\lambda_{1:T,i})$$

This factorization assumes that the latent states $x_{1:T}$ and the assignments $\lambda_{1:T}$ are independent under the approximate posterior.

The individual factors are:
- $q(x_{1:T}) = \prod_{t=1}^T \mathcal{N}(x_t; \mu_t, \Sigma_t)$ is a Gaussian distribution over the latent states
- $q(\lambda_{1:T,i}) = \prod_{t=1}^T q(\lambda_{t,i})$ is a categorical distribution over the assignments

### Evidence Lower Bound (ELBO)

The variational inference objective is to maximize the evidence lower bound (ELBO):

$$\mathcal{L} = \mathbb{E}_{q(x_{1:T}, \lambda_{1:T})}[\log p(y_{1:T}, x_{1:T}, \lambda_{1:T} | u_{1:T}, r_{1:T})] - \mathbb{E}_{q(x_{1:T}, \lambda_{1:T})}[\log q(x_{1:T}, \lambda_{1:T})]$$

Using the factorization of the joint distribution and the variational posterior, the ELBO can be expanded as:

$$\mathcal{L} = \mathbb{E}_{q(x_1)}[\log p(x_1)] - \mathbb{E}_{q(x_1)}[\log q(x_1)] + \sum_{t=1}^{T-1} \mathbb{E}_{q(x_t, x_{t+1})}[\log p(x_{t+1} | x_t, u_t)] - \sum_{t=2}^T \mathbb{E}_{q(x_t)}[\log q(x_t)] + \sum_{t=1}^T \sum_{i=1}^N \mathbb{E}_{q(x_t, \lambda_{t,i})}[\log p(y_{t,i} | x_t, \lambda_{t,i}, r_t)] + \sum_{t=2}^T \sum_{i=1}^N \mathbb{E}_{q(\lambda_{t-1,i}, \lambda_{t,i})}[\log p(\lambda_{t,i} | \lambda_{t-1,i})] - \sum_{t=1}^T \sum_{i=1}^N \mathbb{E}_{q(\lambda_{t,i})}[\log q(\lambda_{t,i})]$$

### Update Equations

The variational parameters are updated using coordinate ascent variational inference (CAVI).

#### Update for $q(x_{1:T})$

The optimal variational distribution for $q(x_{1:T})$ given fixed $q(\lambda_{1:T})$ is obtained by using the Kalman smoothing algorithm with the following modified observation model:

The expected emission matrix at time $t$ is:

$$\bar{C}_t = \sum_{i=1}^N \sum_{\lambda} q(\lambda_{t,i} = \lambda) C_{\lambda}$$

The expected emission covariance at time $t$ is:

$$\bar{\Sigma}_{t,v} = \sum_{i=1}^N \sum_{\lambda} q(\lambda_{t,i} = \lambda) \Sigma_{\lambda}$$

The expected observation at time $t$ is:

$$\bar{y}_t = \sum_{i=1}^N \sum_{\lambda} q(\lambda_{t,i} = \lambda) (y_{t,i} - D_{\lambda}r_t)$$

With these expectations, the Kalman filter forward pass computes:

$$\mu_{t|t-1} = A\mu_{t-1|t-1} + Bu_{t-1}$$
$$\Sigma_{t|t-1} = A\Sigma_{t-1|t-1}A^T + \Sigma_w$$
$$K_t = \Sigma_{t|t-1}\bar{C}_t^T(\bar{C}_t\Sigma_{t|t-1}\bar{C}_t^T + \bar{\Sigma}_{t,v})^{-1}$$
$$\mu_{t|t} = \mu_{t|t-1} + K_t(\bar{y}_t - \bar{C}_t\mu_{t|t-1})$$
$$\Sigma_{t|t} = (I - K_t\bar{C}_t)\Sigma_{t|t-1}$$

And the Kalman smoother backward pass computes:

$$J_t = \Sigma_{t|t}A^T\Sigma_{t+1|t}^{-1}$$
$$\mu_t = \mu_{t|t} + J_t(\mu_{t+1} - \mu_{t+1|t})$$
$$\Sigma_t = \Sigma_{t|t} + J_t(\Sigma_{t+1} - \Sigma_{t+1|t})J_t^T$$

#### Update for $q(\lambda_{1:T,i})$

The optimal variational distribution for each $q(\lambda_{1:T,i})$ given fixed $q(x_{1:T})$ is obtained using the forward-backward algorithm.

The emission potential for observable $i$ at time $t$ with role $\lambda$ is:

$$\phi_{t,i,\lambda} = \exp\left(-\frac{1}{2}(y_{t,i} - \mathbb{E}[C_{\lambda}x_t + D_{\lambda}r_t])^T\Sigma_{\lambda}^{-1}(y_{t,i} - \mathbb{E}[C_{\lambda}x_t + D_{\lambda}r_t]) - \frac{1}{2}\text{Tr}(\Sigma_{\lambda}^{-1}\text{Cov}[C_{\lambda}x_t])\right)$$

where:

$$\mathbb{E}[C_{\lambda}x_t + D_{\lambda}r_t] = C_{\lambda}\mu_t + D_{\lambda}r_t$$
$$\text{Cov}[C_{\lambda}x_t] = C_{\lambda}\Sigma_tC_{\lambda}^T$$

The forward messages $\alpha_{t,i,\lambda}$ are computed as:

$$\alpha_{1,i,\lambda} \propto p(\lambda_{1,i} = \lambda)\phi_{1,i,\lambda}$$
$$\alpha_{t,i,\lambda} \propto \phi_{t,i,\lambda}\sum_{\lambda'} \alpha_{t-1,i,\lambda'}T_{\lambda',\lambda}$$

The backward messages $\beta_{t,i,\lambda}$ are computed as:

$$\beta_{T,i,\lambda} = 1$$
$$\beta_{t,i,\lambda} = \sum_{\lambda'} \beta_{t+1,i,\lambda'}\phi_{t+1,i,\lambda'}T_{\lambda,\lambda'}$$

The updated variational distribution is:

$$q(\lambda_{t,i} = \lambda) \propto \alpha_{t,i,\lambda}\beta_{t,i,\lambda}$$

### Parameter Updates

The model parameters (A, B, C_λ, D_λ, etc.) can be updated using maximum likelihood or Bayesian approaches.

#### Maximum Likelihood Updates

For the transition matrix $A$ and control matrix $B$, the updates are:

$$[A, B] = \left(\sum_{t=1}^{T-1}\mathbb{E}[x_{t+1}][x_t, u_t]^T\right)\left(\sum_{t=1}^{T-1}\mathbb{E}[[x_t, u_t][x_t, u_t]^T]\right)^{-1}$$

For each emission matrix $C_{\lambda}$ and regression matrix $D_{\lambda}$, the updates are:

$$[C_{\lambda}, D_{\lambda}] = \left(\sum_{t=1}^T\sum_{i=1}^N q(\lambda_{t,i} = \lambda)y_{t,i}[x_t, r_t]^T\right)\left(\sum_{t=1}^T\sum_{i=1}^N q(\lambda_{t,i} = \lambda)\mathbb{E}[[x_t, r_t][x_t, r_t]^T]\right)^{-1}$$

#### Bayesian Updates

In the Bayesian setting, conjugate priors are used for the parameters:

- Matrix Normal-Wishart for $[A, B]$ and $[C_{\lambda}, D_{\lambda}]$
- Dirichlet for transition probabilities $T_{\lambda,\lambda'}$

The posterior update equations follow the standard conjugate update rules with expectations taken with respect to the variational distribution.

## Multiple Objects Extension

For multiple objects, the latent state is partitioned as:

$$x_t = [s_t, b_{t,1}, z_{t,1}, b_{t,2}, z_{t,2}, \ldots, b_{t,K}, z_{t,K}]$$

where:
- $s_t$ is the shared environment latent
- $b_{t,k}$ is the boundary latent for object $k$
- $z_{t,k}$ is the internal latent for object $k$
- $K$ is the number of objects

The transition matrix $A$ has a block structure that enforces no direct interactions between different objects or between objects and the environment:

$$A = \begin{pmatrix}
A_{ss} & A_{sb_1} & 0 & A_{sb_2} & 0 & \ldots & A_{sb_K} & 0 \\
A_{b_1s} & A_{b_1b_1} & A_{b_1z_1} & 0 & 0 & \ldots & 0 & 0 \\
0 & A_{z_1b_1} & A_{z_1z_1} & 0 & 0 & \ldots & 0 & 0 \\
A_{b_2s} & 0 & 0 & A_{b_2b_2} & A_{b_2z_2} & \ldots & 0 & 0 \\
0 & 0 & 0 & A_{z_2b_2} & A_{z_2z_2} & \ldots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
A_{b_Ks} & 0 & 0 & 0 & 0 & \ldots & A_{b_Kb_K} & A_{b_Kz_K} \\
0 & 0 & 0 & 0 & 0 & \ldots & A_{z_Kb_K} & A_{z_Kz_K}
\end{pmatrix}$$

The roles and transitions are also modified to accommodate multiple objects, with each observable potentially assigned to any object's boundary or internal state.

## Theoretical Properties

### Markov Blanket Criteria

The model enforces the Markov blanket criteria, which can be formally stated as:

$$p(z_{t+1} | z_t, b_t, s_t) = p(z_{t+1} | z_t, b_t)$$
$$p(s_{t+1} | z_t, b_t, s_t) = p(s_{t+1} | b_t, s_t)$$

These conditional independence relations are enforced by the structure of the transition matrix $A$.

### Relationship to Free Energy Principle

The DMBD algorithm is related to the Free Energy Principle and active inference frameworks in computational neuroscience. According to these frameworks, biological systems act to minimize their variational free energy (negative ELBO) to maintain their integrity and avoid surprising states.

The Markov blanket structure identified by DMBD corresponds to the statistical separation between an organism (object) and its environment, with the boundary mediating their interactions.

### Information-Theoretic Interpretation

From an information-theoretic perspective, the Markov blanket structure minimizes the information exchange between the object and environment, subject to the constraint that the object maintains a model of its relevant environment through the boundary.

The mutual information between object and environment conditioned on the boundary is zero:

$$I(s_t; z_t | b_t) = 0$$

This information minimization is consistent with theories of cognitive parsimony and efficient coding in biological systems.

## Implementation Details

### Matrix Masking

To enforce the Markov blanket structure constraints on the parameters, DMBD uses masking. For example, the transition matrix $A$ is masked as:

$$A_{\text{masked}} = A \odot M_A$$

where $\odot$ is element-wise multiplication and $M_A$ is a binary mask with zeros in the positions where $A$ should have zeros to enforce the Markov blanket structure.

### Role-Based Masking

Similarly, the emission matrices $C_{\lambda}$ are masked based on their role:

$$C_{\lambda,\text{masked}} = C_{\lambda} \odot M_{\lambda}$$

where $M_{\lambda}$ is a binary mask with the appropriate structure for role $\lambda$.

### Numerical Stability

For numerical stability, DMBD uses several techniques:
- Logarithmic computations for the forward-backward algorithm
- Stabilized Kalman filter updates
- Gradient clipping for parameter updates
- Temperature annealing for assignment distributions

## References

1. Friston, K. J. (2019). A free energy principle for a particular physics. arXiv preprint arXiv:1906.10184.
2. Pearl, J. (1988). Probabilistic reasoning in intelligent systems: Networks of plausible inference. Morgan Kaufmann.
3. Zhu, J., Chen, N., & Xing, E. P. (2014). Bayesian inference with posterior regularization and applications to infinite latent SVMs. The Journal of Machine Learning Research, 15(1), 1799-1847.
4. Beal, M. J. (2003). Variational algorithms for approximate Bayesian inference. University of London, University College London.
5. Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear dynamical systems. Technical Report CRG-TR-96-2, University of Toronto, Dept. of Computer Science. 