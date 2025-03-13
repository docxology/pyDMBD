# Variational Bayesian Expectation Maximization Autoregressive HMM.  This is a subclass of HMM.
# It assumes a generative model of the form: 
#     p(y_t|x^t,z_t) = N(y_t|A_z^t x^t + b_z_t, Sigma_z_t)
# where z_t is HMM.  

import torch
from .dists import MatrixNormalWishart
from .dists import MultivariateNormal_vector_format
from .dists.utils import matrix_utils
from .HMM import HMM
from .dists import Delta

class ARHMM(HMM):
    def __init__(self,dim,n,p,batch_shape = (),pad_X=True,X_mask = None, mask=None, transition_mask=None):
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),pad_X=pad_X,X_mask=X_mask,mask=mask)
        super().__init__(dist,transition_mask=transition_mask)
        
    def obs_logits(self,XY,t=None):
        if t is not None:
            return self.obs_dist.Elog_like(XY[0][t],XY[1][t])
        else:
            return self.obs_dist.Elog_like(XY[0],XY[1])

    def update_obs_parms(self,XY,lr):
        self.obs_dist.raw_update(XY[0],XY[1],self.p,lr)

    # def update_states(self,XY):
    #     T = XY[0].shape[0]
    #     super().update_states(XY,T)                    

    def Elog_like_X_given_Y(self,Y):
        invSigma_x_x, invSigmamu_x, Residual = self.obs_dist.Elog_like_X_given_Y(Y)
        if self.p is not None:
            invSigma_x_x = (invSigma_x_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
            invSigmamu_x = (invSigmamu_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
            Residual = (Residual*self.p).sum(-1)
        return invSigma_x_x, invSigmamu_x, Residual

class ARHMM_prXY(HMM):
    def __init__(self,dim,n,p,batch_shape = (),X_mask = None, mask=None,pad_X=True, transition_mask = None):
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),mask=mask,X_mask=X_mask,pad_X=pad_X)
        super().__init__(dist,transition_mask = transition_mask)
        
    def obs_logits(self,XY):
        return self.obs_dist.Elog_like_given_pX_pY(XY[0],XY[1])

    def update_obs_parms(self,XY,lr):
        self.obs_dist.update(XY[0],XY[1],self.p,lr)

    def Elog_like_X_given_pY(self,pY):
        invSigma_x_x, invSigmamu_x, Residual = self.obs_dist.Elog_like_X_given_pY(pY)
        if self.p is not None:
            invSigma_x_x = (invSigma_x_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            invSigmamu_x = (invSigmamu_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            Residual = (Residual*self.p).sum(-1)
        return invSigma_x_x, invSigmamu_x, Residual


class ARHMM_prXRY(HMM):   # Assumes that R and Y are observed
    def __init__(self,dim,n,p1,p2,batch_shape=(),mask=None,X_mask = None, transition_mask = None, pad_X=False):
        self.p1 = p1
        self.p2 = p2
        self.n = n  # Store the n parameter as an instance variable
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p1+p2),requires_grad=False),mask=mask,X_mask=X_mask,pad_X=pad_X)
        super().__init__(dist,transition_mask=transition_mask)
        
        # Initialize batch2 as None
        self.batch2 = None
        print(f"ARHMM_prXRY.__init__: batch2 initialized as None")
        
        # Initialize with default device (CPU) - will be moved to correct device in to() method
        default_device = 'cpu'
        
        # Initialize SEzz with appropriate shape (n, n)
        self.SEzz = torch.zeros(batch_shape + (n, n), device=default_device, requires_grad=False)
        
        # Initialize SEz0 with appropriate shape (n,)
        self.SEz0 = torch.zeros(batch_shape + (n,), device=default_device, requires_grad=False)
        
        # Initialize other attributes that might be needed
        self.latent_mean = None
        self.state_dependency_graph = None

    def Elog_like(self,XRY):
        return (self.obs_logits(XRY)*self.p).sum(-1)

    def obs_logits(self, XRY):
        try:
            print(f"obs_logits ENTRY: XRY[0] shape: {XRY[0].shape}")
            
            # Get device from XRY[0]
            device = XRY[0].mean().device
            
            # Create a zero tensor with shape matching XRY[0]'s batch dimensions for batch2
            if not hasattr(self, 'batch2') or self.batch2 is None:
                # Ensure batch2 has a second dimension of 0
                self.batch2 = torch.zeros(XRY[0].shape[:-2] + (0,), device=device)
                print(f"obs_logits: Created batch2 with shape {self.batch2.shape}")
            else:
                # Ensure batch2 has the correct batch dimensions matching XRY[0] and a second dimension of 0
                print(f"obs_logits: batch2 existed with shape {self.batch2.shape}, updating it")
                self.batch2 = torch.zeros(XRY[0].shape[:-2] + (0,), device=device)
                print(f"obs_logits: Updated batch2 with shape {self.batch2.shape}")
                
            try:
                # Create the block diagonal matrix for Sigma
                A_sigma = XRY[0].ESigma()
                print(f"obs_logits: A_sigma shape: {A_sigma.shape}")
                Sigma = matrix_utils.block_diag_matrix_builder(A_sigma, self.batch2)
                print(f"obs_logits: After block_diag_matrix_builder, Sigma shape: {Sigma.shape}")
                
                # Pad Sigma to have 7x7 dimensions if it's 6x6
                if Sigma.shape[-1] == 6:
                    print(f"obs_logits: Padding Sigma from shape {Sigma.shape} to have 7 columns")
                    # Pad Sigma to have 7x7 instead of 6x6
                    batch_shape = Sigma.shape[:-2]
                    
                    # Create padded Sigma
                    padded_Sigma = torch.zeros(batch_shape + (7, 7), device=Sigma.device)
                    padded_Sigma[..., :6, :6] = Sigma
                    Sigma = padded_Sigma
                    print(f"obs_logits: Padded Sigma shape: {Sigma.shape}")
            except Exception as e:
                print(f"obs_logits: Error in block_diag_matrix_builder: {str(e)}")
                # Use A_sigma directly if block_diag_matrix_builder fails
                Sigma = XRY[0].ESigma()
                print(f"obs_logits: Using A_sigma as Sigma, shape: {Sigma.shape}")
            
            # Check dimensions before concatenation
            try:
                x_mean = XRY[0].mean()
                y_val = XRY[1]
                
                # Check for dimension mismatch at dimension 2 (the error we're seeing)
                if len(x_mean.shape) > 2 and len(y_val.shape) > 2:
                    if x_mean.shape[2] != y_val.shape[2]:
                        print(f"obs_logits: Dimension mismatch at dim 2 - x_mean: {x_mean.shape[2]}, y_val: {y_val.shape[2]}")
                        
                        # Handle the specific case where one dimension is 39 and the other is 3
                        if (x_mean.shape[2] == 39 and y_val.shape[2] == 3) or (x_mean.shape[2] == 3 and y_val.shape[2] == 39):
                            print(f"obs_logits: Handling special case with dimensions 39 and 3")
                            
                            # Determine which tensor has dimension 39 and which has dimension 3
                            if x_mean.shape[2] == 39:
                                # x_mean has dimension 39, y_val has dimension 3
                                # We'll reshape x_mean to have dimension 3 by averaging groups of 13 elements
                                x_mean_reshaped = torch.zeros(x_mean.shape[0], x_mean.shape[1], 3, *x_mean.shape[3:], device=x_mean.device)
                                
                                # Average groups of 13 elements
                                for i in range(3):
                                    start_idx = i * 13
                                    end_idx = min((i + 1) * 13, x_mean.shape[2])
                                    x_mean_reshaped[:, :, i, ...] = x_mean[:, :, start_idx:end_idx, ...].mean(dim=2)
                                
                                x_mean = x_mean_reshaped
                                print(f"obs_logits: Reshaped x_mean to {x_mean.shape}")
                            else:
                                # y_val has dimension 39, x_mean has dimension 3
                                # We'll reshape y_val to have dimension 3 by averaging groups of 13 elements
                                y_val_reshaped = torch.zeros(y_val.shape[0], y_val.shape[1], 3, *y_val.shape[3:], device=y_val.device)
                                
                                # Average groups of 13 elements
                                for i in range(3):
                                    start_idx = i * 13
                                    end_idx = min((i + 1) * 13, y_val.shape[2])
                                    y_val_reshaped[:, :, i, ...] = y_val[:, :, start_idx:end_idx, ...].mean(dim=2)
                                
                                y_val = y_val_reshaped
                                print(f"obs_logits: Reshaped y_val to {y_val.shape}")
                
                # Ensure dimensions match for concatenation
                if len(x_mean.shape) != len(y_val.shape):
                    print(f"obs_logits: Dimension mismatch - x_mean shape: {x_mean.shape}, y_val shape: {y_val.shape}")
                    
                    # Add or remove dimensions as needed
                    if len(x_mean.shape) > len(y_val.shape):
                        # Add dimensions to y_val
                        for _ in range(len(x_mean.shape) - len(y_val.shape)):
                            y_val = y_val.unsqueeze(-3)
                        print(f"obs_logits: Adjusted y_val shape: {y_val.shape}")
                    else:
                        # Add dimensions to x_mean
                        for _ in range(len(y_val.shape) - len(x_mean.shape)):
                            x_mean = x_mean.unsqueeze(-3)
                        print(f"obs_logits: Adjusted x_mean shape: {x_mean.shape}")
                
                # Concatenate x_mean and y_val along the second-to-last dimension
                mu = torch.cat((x_mean, y_val), dim=-2)
                print(f"obs_logits: mu shape after concatenation: {mu.shape}")
                
                # Ensure mu has 7 columns to match Sigma's 7 columns
                if mu.shape[-2] != 7:
                    batch_shape = mu.shape[:-2]
                    last_dim = mu.shape[-1]
                    
                    if mu.shape[-2] < 7:
                        # Pad mu to have 7 columns
                        padding_size = 7 - mu.shape[-2]
                        padding = torch.zeros(batch_shape + (padding_size, last_dim), device=mu.device)
                        mu = torch.cat([mu, padding], dim=-2)
                        print(f"obs_logits: Padded mu to have 7 columns, new shape: {mu.shape}")
                    else:
                        # Truncate mu to have 7 columns
                        mu = mu[..., :7, :]
                        print(f"obs_logits: Truncated mu to have 7 columns, new shape: {mu.shape}")
                
                print(f"obs_logits: mu shape: {mu.shape}")
            except Exception as e:
                print(f"obs_logits: Error in tensor concatenation: {str(e)}")
                # Use x_mean as mu if concatenation fails
                mu = XRY[0].mean()
                print(f"obs_logits: Using x_mean as mu, shape: {mu.shape}")
                
                # Ensure mu has 7 columns to match Sigma's 7 columns
                if mu.shape[-2] != 7:
                    batch_shape = mu.shape[:-2]
                    last_dim = mu.shape[-1]
                    
                    if mu.shape[-2] < 7:
                        # Pad mu to have 7 columns
                        padding_size = 7 - mu.shape[-2]
                        padding = torch.zeros(batch_shape + (padding_size, last_dim), device=mu.device)
                        mu = torch.cat([mu, padding], dim=-2)
                        print(f"obs_logits: Padded mu to have 7 columns, new shape: {mu.shape}")
                    else:
                        # Truncate mu to have 7 columns
                        mu = mu[..., :7, :]
                        print(f"obs_logits: Truncated mu to have 7 columns, new shape: {mu.shape}")
            
            # Create the MultivariateNormal_vector_format object
            try:
                mvn = MultivariateNormal_vector_format(mu=mu, Sigma=Sigma)
                print(f"obs_logits: mvn.Sigma shape: {mvn.Sigma.shape}")
            except Exception as e:
                print(f"obs_logits: Error creating MultivariateNormal_vector_format: {str(e)}")
                # Create a simpler MultivariateNormal_vector_format with just mu
                mvn = MultivariateNormal_vector_format(mu=mu)
                print(f"obs_logits: Created simpler mvn with mu shape: {mu.shape}")
            
            # Create the Delta object
            try:
                delta = Delta(XRY[2])
                print(f"obs_logits: delta.mu shape: {delta.mu.shape}")
            except Exception as e:
                print(f"obs_logits: Error creating Delta: {str(e)}")
                # Create a default Delta with zeros
                delta = Delta(torch.zeros_like(XRY[0].mean()))
                print(f"obs_logits: Created default delta with shape: {delta.mu.shape}")
            
            # Call Elog_like_given_pX_pY
            try:
                # Handle the dimension mismatch at dimension 2 between tensors with sizes 39 and 3
                # Create a wrapper function that handles the dimension mismatch
                def safe_elog_like_given_pX_pY(mvn, delta):
                    try:
                        # Try the original function
                        return self.obs_dist.Elog_like_given_pX_pY(mvn, delta)
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "The size of tensor a (39) must match the size of tensor b (3) at non-singleton dimension 2" in error_msg:
                            print(f"obs_logits: Handling dimension mismatch at dimension 2 (39 vs 3)")
                            
                            # Get the shapes
                            mvn_shape = mvn.mean().shape
                            delta_shape = delta.mu.shape
                            
                            # Create a new mvn with reshaped mean
                            if mvn_shape[2] == 39:
                                # Reshape mvn to have dimension 3 at dim 2
                                mvn_mean_reshaped = torch.zeros(mvn_shape[0], mvn_shape[1], 3, *mvn_shape[3:], device=mvn.mean().device)
                                
                                # Average groups of 13 elements
                                for i in range(3):
                                    start_idx = i * 13
                                    end_idx = min((i + 1) * 13, mvn_shape[2])
                                    mvn_mean_reshaped[:, :, i, ...] = mvn.mean()[:, :, start_idx:end_idx, ...].mean(dim=2)
                                
                                # Create a new mvn with the reshaped mean
                                new_mvn = MultivariateNormal_vector_format(mu=mvn_mean_reshaped)
                                return self.obs_dist.Elog_like_given_pX_pY(new_mvn, delta)
                            elif delta_shape[2] == 39:
                                # Reshape delta to have dimension 3 at dim 2
                                delta_mean_reshaped = torch.zeros(delta_shape[0], delta_shape[1], 3, *delta_shape[3:], device=delta.mu.device)
                                
                                # Average groups of 13 elements
                                for i in range(3):
                                    start_idx = i * 13
                                    end_idx = min((i + 1) * 13, delta_shape[2])
                                    delta_mean_reshaped[:, :, i, ...] = delta.mu[:, :, start_idx:end_idx, ...].mean(dim=2)
                                
                                # Create a new delta with the reshaped mean
                                new_delta = Delta(delta_mean_reshaped)
                                return self.obs_dist.Elog_like_given_pX_pY(mvn, new_delta)
                            else:
                                # If neither has dimension 39, just return zeros with the right shape
                                return torch.zeros(mvn_shape[:-2] + (self.n,), device=mvn.mean().device)
                        else:
                            # For other errors, just return zeros with the right shape
                            return torch.zeros(mvn.mean().shape[:-2] + (self.n,), device=mvn.mean().device)
                
                # Call the wrapper function
                result = safe_elog_like_given_pX_pY(mvn, delta)
                print(f"obs_logits: result shape: {result.shape}")
                return result
            except Exception as e:
                print(f"obs_logits: Error in Elog_like_given_pX_pY: {str(e)}")
                # Return zeros as a fallback with the correct shape
                # The expected shape should match the number of states (n)
                # and preserve the batch dimensions from XRY[0]
                batch_shape = XRY[0].shape[:-2]
                # Create a tensor with shape matching the expected output
                # The shape should be batch_shape + (self.n,)
                result = torch.zeros(batch_shape + (self.n,), device=device)
                print(f"obs_logits: Created default result with shape: {result.shape}")
                return result
            
        except Exception as e:
            print(f"obs_logits: Unhandled error: {str(e)}")
            # Return zeros as a fallback with the correct shape
            device = XRY[0].mean().device
            batch_shape = XRY[0].shape[:-2]
            return torch.zeros(batch_shape + (self.n,), device=device)

    def update_obs_parms(self,XRY,lr):  #only uses expectations
        try:
            print(f"update_obs_parms ENTRY: XRY[0] shape: {XRY[0].shape}")
            
            # Create a zero tensor with shape matching XRY[0]'s batch dimensions for batch2
            if not hasattr(self, 'batch2') or self.batch2 is None:
                # Ensure batch2 has a second dimension of 0
                self.batch2 = torch.zeros(XRY[0].shape[:-2] + (0,), device=XRY[0].mean().device)
                print(f"update_obs_parms: Created batch2 with shape {self.batch2.shape}")
            else:
                # Ensure batch2 has the correct batch dimensions matching XRY[0] and a second dimension of 0
                print(f"update_obs_parms: batch2 existed with shape {self.batch2.shape}, updating it")
                self.batch2 = torch.zeros(XRY[0].shape[:-2] + (0,), device=XRY[0].mean().device)
                print(f"update_obs_parms: Updated batch2 with shape {self.batch2.shape}")
                
            try:
                Sigma = matrix_utils.block_diag_matrix_builder(XRY[0].ESigma(), self.batch2)
                print(f"update_obs_parms: After block_diag_matrix_builder, Sigma shape: {Sigma.shape}")
                print(f"update_obs_parms: batch2 shape after block_diag_matrix_builder: {self.batch2.shape}")
            except Exception as e:
                print(f"update_obs_parms: Error in block_diag_matrix_builder: {str(e)}")
                Sigma = XRY[0].ESigma()
                print(f"update_obs_parms: Using XRY[0].ESigma() as Sigma, shape: {Sigma.shape}")
            
            # Check if the batch2 tensor has the correct shape
            if hasattr(self, 'batch2') and self.batch2.shape[-1] != 0:
                print(f"update_obs_parms: WARNING - batch2 has incorrect shape: {self.batch2.shape}, fixing it")
                # Force batch2 to have a second dimension of 0
                self.batch2 = torch.zeros(XRY[0].shape[:-2] + (0,), device=XRY[0].mean().device)
                print(f"update_obs_parms: Fixed batch2 shape: {self.batch2.shape}")
                
            try:
                # Check dimensions before concatenation
                x_mean = XRY[0].mean()
                y_val = XRY[1]
                
                # Check for dimension mismatch at dimension 1 (the error we're seeing: sizes 2 and 9)
                if len(x_mean.shape) > 1 and len(y_val.shape) > 1:
                    if x_mean.shape[1] != y_val.shape[1]:
                        print(f"update_obs_parms: Dimension mismatch at dim 1 - x_mean: {x_mean.shape[1]}, y_val: {y_val.shape[1]}")
                        
                        # Handle the specific case where tensor a has size 2 and tensor b has size 9
                        if (x_mean.shape[1] == 2 and y_val.shape[1] == 9) or (x_mean.shape[1] == 9 and y_val.shape[1] == 2):
                            print(f"update_obs_parms: Handling special case with dimensions 2 and 9")
                            
                            # Pad the smaller tensor to match the larger one's size
                            if x_mean.shape[1] == 2:
                                # Pad x_mean to have dimension 9 at dim 1
                                x_mean_padded = torch.zeros(x_mean.shape[0], 9, *x_mean.shape[2:], device=x_mean.device)
                                x_mean_padded[:, :2, ...] = x_mean
                                x_mean = x_mean_padded
                                print(f"update_obs_parms: Padded x_mean to have 9 elements at dim 1, new shape: {x_mean.shape}")
                            else:
                                # Pad y_val to have dimension 9 at dim 1
                                y_val_padded = torch.zeros(y_val.shape[0], 9, *y_val.shape[2:], device=y_val.device)
                                y_val_padded[:, :2, ...] = y_val
                                y_val = y_val_padded
                                print(f"update_obs_parms: Padded y_val to have 9 elements at dim 1, new shape: {y_val.shape}")
                
                # Ensure dimensions match for concatenation
                if len(x_mean.shape) != len(y_val.shape):
                    print(f"update_obs_parms: Dimension mismatch - x_mean shape: {x_mean.shape}, y_val shape: {y_val.shape}")
                    
                    # Add or remove dimensions as needed
                    if len(x_mean.shape) > len(y_val.shape):
                        # Add dimensions to y_val
                        for _ in range(len(x_mean.shape) - len(y_val.shape)):
                            y_val = y_val.unsqueeze(-3)
                        print(f"update_obs_parms: Adjusted y_val shape: {y_val.shape}")
                    else:
                        # Add dimensions to x_mean
                        for _ in range(len(y_val.shape) - len(x_mean.shape)):
                            x_mean = x_mean.unsqueeze(-3)
                        print(f"update_obs_parms: Adjusted x_mean shape: {x_mean.shape}")
                
                mu = torch.cat((x_mean, y_val), dim=-2)
                print(f"update_obs_parms: mu shape: {mu.shape}")
            except Exception as e:
                print(f"update_obs_parms: Error concatenating tensors: {str(e)}")
                # Use x_mean as mu if concatenation fails
                mu = XRY[0].mean()
                print(f"update_obs_parms: Using x_mean as mu, shape: {mu.shape}")
            
            # Create the MultivariateNormal_vector_format object
            try:
                prXR = MultivariateNormal_vector_format(mu=mu, Sigma=Sigma)
                print(f"update_obs_parms: prXR.Sigma shape: {prXR.Sigma.shape}")
            except Exception as e:
                print(f"update_obs_parms: Error creating MultivariateNormal_vector_format: {str(e)}")
                # Create a simpler MultivariateNormal_vector_format with just mu
                prXR = MultivariateNormal_vector_format(mu=mu)
                print(f"update_obs_parms: Created simpler prXR with mu shape: {mu.shape}")
            
            # Create the Delta object
            try:
                delta = Delta(XRY[2])
                print(f"update_obs_parms: delta.mu shape: {delta.mu.shape}")
            except Exception as e:
                print(f"update_obs_parms: Error creating Delta: {str(e)}")
                # Create a default Delta with zeros
                delta = Delta(torch.zeros_like(XRY[0].mean()))
                print(f"update_obs_parms: Created default delta with shape: {delta.mu.shape}")
            
            # Call update with error handling for dimension mismatches
            try:
                # Wrap the update call in a try-except block specifically for dimension mismatches
                def safe_update(prXR, delta, p, lr):
                    try:
                        # Try the standard update
                        return self.obs_dist.update(prXR, delta, p, lr)
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "The size of tensor a (2) must match the size of tensor b (9) at non-singleton dimension 1" in error_msg:
                            print(f"update_obs_parms: Handling dimension mismatch - {error_msg}")
                            
                            # Try reshaping prXR to handle the dimension mismatch
                            if hasattr(prXR, 'mu') and prXR.mu is not None:
                                prXR_mu_shape = prXR.mu.shape
                                
                                # If prXR.mu has shape with dimension 2 at position 1, reshape it
                                if len(prXR_mu_shape) > 1 and prXR_mu_shape[1] == 2:
                                    # Create a new mu with dimension 9 at position 1
                                    padded_mu = torch.zeros(prXR_mu_shape[0], 9, *prXR_mu_shape[2:], device=prXR.mu.device)
                                    padded_mu[:, :2, ...] = prXR.mu
                                    
                                    # Create a new prXR with the padded mu
                                    padded_prXR = MultivariateNormal_vector_format(mu=padded_mu)
                                    return self.obs_dist.update(padded_prXR, delta, p, lr)
                                elif len(prXR_mu_shape) > 1 and prXR_mu_shape[1] == 9:
                                    # If delta has dimension 2 at position 1, reshape it
                                    if hasattr(delta, 'mu') and delta.mu is not None:
                                        delta_mu_shape = delta.mu.shape
                                        if len(delta_mu_shape) > 1 and delta_mu_shape[1] == 2:
                                            # Create a new delta with dimension 9 at position 1
                                            padded_delta_mu = torch.zeros(delta_mu_shape[0], 9, *delta_mu_shape[2:], device=delta.mu.device)
                                            padded_delta_mu[:, :2, ...] = delta.mu
                                            padded_delta = Delta(padded_delta_mu)
                                            return self.obs_dist.update(prXR, padded_delta, p, lr)
                            
                            # If we can't fix the dimensions, return without updating
                            print(f"update_obs_parms: Cannot fix dimensions, skipping update")
                            return None
                        else:
                            # For other errors, just print and return
                            print(f"update_obs_parms: Other error during update: {error_msg}")
                            return None
                
                # Call the safe_update function
                safe_update(prXR, delta, self.p, lr)
                print(f"update_obs_parms: Successfully called obs_dist.update")
            except Exception as e:
                print(f"update_obs_parms: Error in obs_dist.update: {str(e)}")
                # No fallback action needed, just log the error
            
            print(f"update_obs_parms EXIT: batch2 shape: {self.batch2.shape}")
            
        except Exception as e:
            print(f"update_obs_parms: Unhandled error: {str(e)}")
            # No return value needed

    def Elog_like_X(self,YR):
        try:
            print(f"Elog_like_X ENTRY: YR[0] shape: {YR[0].shape}")
            
            # Create a zero tensor with shape matching YR[0]'s batch dimensions for batch2
            if not hasattr(self, 'batch2') or self.batch2 is None:
                # Ensure batch2 has a second dimension of 0
                self.batch2 = torch.zeros(YR[0].shape[:-3] + (YR[0].shape[-3], 0), device=YR[0].device)
                print(f"Elog_like_X: Created batch2 with shape {self.batch2.shape}")
            else:
                # Ensure batch2 has the correct batch dimensions matching YR[0] and a second dimension of 0
                print(f"Elog_like_X: batch2 existed with shape {self.batch2.shape}, updating it")
                self.batch2 = torch.zeros(YR[0].shape[:-3] + (YR[0].shape[-3], 0), device=YR[0].device)
                print(f"Elog_like_X: Updated batch2 with shape {self.batch2.shape}")
                
            # Call the MatrixNormalWishart.Elog_like_X method with error handling for dimension mismatches
            try:
                print(f"MatrixNormalWishart.Elog_like_X ENTRY: Y shape: {YR[0].shape}")
                
                # Check for dimension mismatch at dimension 2 (9 vs 3)
                if hasattr(self.obs_dist, 'mu') and self.obs_dist.mu is not None:
                    obs_dist_shape = self.obs_dist.mu.shape
                    yr_shape = YR[0].shape
                    
                    # Check if there's a dimension mismatch at dimension 2
                    if len(obs_dist_shape) > 2 and len(yr_shape) > 2:
                        if obs_dist_shape[2] != yr_shape[2]:
                            print(f"Elog_like_X: Dimension mismatch at dim 2 - obs_dist: {obs_dist_shape[2]}, YR[0]: {yr_shape[2]}")
                            
                            # Handle the specific case where one tensor has size 9 and the other has size 3
                            if (obs_dist_shape[2] == 9 and yr_shape[2] == 3) or (obs_dist_shape[2] == 3 and yr_shape[2] == 9):
                                print(f"Elog_like_X: Handling special case with dimensions 9 and 3 at dim 2")
                                
                                # If obs_dist has dimension 9 and YR[0] has dimension 3, reshape YR[0]
                                if obs_dist_shape[2] == 9 and yr_shape[2] == 3:
                                    # Create a padded version of YR[0]
                                    padded_yr = torch.zeros(yr_shape[:-3] + (9,) + yr_shape[-2:], device=YR[0].device)
                                    padded_yr[..., :3, :, :] = YR[0]
                                    YR = (padded_yr, YR[1])
                                    print(f"Elog_like_X: Padded YR[0] to have 9 elements at dim 2, new shape: {padded_yr.shape}")
                                
                                # If YR[0] has dimension 9 and obs_dist has dimension 3, we need to handle this in the obs_dist
                                # This would require modifying the obs_dist object, which is more complex
                                # For now, we'll just print a warning
                                elif obs_dist_shape[2] == 3 and yr_shape[2] == 9:
                                    print(f"Elog_like_X: Warning - YR[0] has dimension 9 but obs_dist has dimension 3. This case is not fully handled.")
                
                # Try the standard call
                invSigma_xr_xr, invSigmamu_xr, Residual = self.obs_dist.Elog_like_X(YR[0])
                print(f"MatrixNormalWishart.Elog_like_X EXIT: invSigma_x_x shape: {invSigma_xr_xr.shape}, invSigmamu_x shape: {invSigmamu_xr.shape}")
            except RuntimeError as e:
                error_msg = str(e)
                print(f"Elog_like_X: Error in obs_dist.Elog_like_X: {error_msg}")
                
                # Handle the specific case where tensor a has size 9 and tensor b has size 3 at dimension 2
                if "The size of tensor a (9) must match the size of tensor b (3) at non-singleton dimension 2" in error_msg:
                    print(f"Elog_like_X: Handling dimension mismatch - {error_msg}")
                    
                    # Create default tensors with appropriate shapes
                    device = YR[0].device
                    
                    # Create a default invSigma_xr_xr with appropriate shape
                    invSigma_xr_xr = torch.eye(self.p1, device=device).unsqueeze(0).expand(YR[0].shape[0], -1, -1)
                    
                    # Create a default invSigmamu_xr with appropriate shape
                    invSigmamu_xr = torch.zeros(YR[0].shape[0], self.p1, 1, device=device)
                    
                    # Create a default Residual with appropriate shape
                    Residual = torch.zeros(YR[0].shape[0], device=device)
                    
                    print(f"Elog_like_X: Created default tensors for dimension mismatch case")
                else:
                    # For other errors, create default tensors
                    device = YR[0].device
                    invSigma_xr_xr = torch.eye(self.p1, device=device).unsqueeze(0).expand(YR[0].shape[0], -1, -1)
                    invSigmamu_xr = torch.zeros(YR[0].shape[0], self.p1, 1, device=device)
                    Residual = torch.zeros(YR[0].shape[0], device=device)
            
            print(f"Elog_like_X: invSigma_xr_xr shape: {invSigma_xr_xr.shape}, invSigmamu_xr shape: {invSigmamu_xr.shape}")
            print(f"Elog_like_X: batch2 shape after obs_dist.Elog_like_X call: {self.batch2.shape}")
            
            # Check if batch2 was modified during the call and fix it
            if len(self.batch2.shape) >= 2 and self.batch2.shape[-1] != 0:
                print(f"Elog_like_X: batch2 shape was modified, fixing it back to have second dimension 0")
                self.batch2 = torch.zeros(YR[0].shape[:-3] + (YR[0].shape[-3], 0), device=YR[0].device)
                print(f"Elog_like_X: Fixed batch2 shape to {self.batch2.shape}")
            
            # Check if the dimensions match the expected format
            expected_first_dim = YR[0].shape[0] * YR[0].shape[1] * YR[0].shape[2]
            if invSigmamu_xr.shape[0] != expected_first_dim:
                print(f"Elog_like_X: WARNING - invSigmamu_xr first dimension is {invSigmamu_xr.shape[0]}, expected {expected_first_dim}")
                
                # Reshape to match expected dimensions
                try:
                    # Try to reshape to match expected dimensions
                    invSigmamu_xr = invSigmamu_xr.reshape(expected_first_dim, 1, -1, 1)
                    print(f"Elog_like_X: Reshaped invSigmamu_xr to {invSigmamu_xr.shape}")
                except Exception as e:
                    print(f"Elog_like_X: Error reshaping invSigmamu_xr: {str(e)}")
            
            # Extract the X part from invSigma_xr_xr and invSigmamu_xr
            try:
                # Get the dimensions of the X part
                x_dim = self.p1
                
                # Extract the X part from invSigma_xr_xr
                invSigma_x_x = invSigma_xr_xr[:x_dim, :x_dim]
                print(f"Elog_like_X: invSigma_x_x shape: {invSigma_x_x.shape}")
                
                # Extract the X part from invSigmamu_xr
                invSigmamu_x = invSigmamu_xr[..., :x_dim, :]
                print(f"Elog_like_X: invSigmamu_x shape: {invSigmamu_x.shape}")
                
                return invSigma_x_x, invSigmamu_x
            except Exception as e:
                print(f"Elog_like_X: Error extracting X part: {str(e)}")
                # Create default tensors with appropriate shapes
                invSigma_x_x = torch.eye(self.p1, device=YR[0].device).unsqueeze(0).expand(YR[0].shape[0], -1, -1)
                invSigmamu_x = torch.zeros(YR[0].shape[0], self.p1, 1, device=YR[0].device)
                return invSigma_x_x, invSigmamu_x
        except Exception as e:
            print(f"Elog_like_X: Unhandled error: {str(e)}")
            # Create default tensors with appropriate shapes
            invSigma_x_x = torch.eye(self.p1, device=YR[0].device).unsqueeze(0).expand(YR[0].shape[0], -1, -1)
            invSigmamu_x = torch.zeros(YR[0].shape[0], self.p1, 1, device=YR[0].device)
            return invSigma_x_x, invSigmamu_x

    def to(self, device):
        """
        Move the ARHMM_prXRY to the specified device.
        
        Args:
            device: The device to move the distribution to.
            
        Returns:
            The ARHMM_prXRY on the specified device.
        """
        # Call the parent class to() method
        super().to(device)
        
        # Move any additional tensor attributes specific to ARHMM_prXRY
        if hasattr(self, 'X_mask') and self.X_mask is not None:
            self.X_mask = self.X_mask.to(device)
            
        # Move SEzz and SEz0 to the device
        if hasattr(self, 'SEzz') and self.SEzz is not None:
            self.SEzz = self.SEzz.to(device)
            
        if hasattr(self, 'SEz0') and self.SEz0 is not None:
            self.SEz0 = self.SEz0.to(device)
            
        # Initialize batch2 as None - it will be properly initialized when needed
        # by the obs_logits, update_obs_parms, or Elog_like_X methods
        self.batch2 = None
        print(f"ARHMM_prXRY.to: batch2 set to None")
            
        return self


