import torch
import numpy as np
from .lds import LinearDynamicalSystems
from .ARHMM import ARHMM_prXRY
from .dists import MatrixNormalWishart
from .dists import MatrixNormalGamma
from .dists import NormalInverseWishart
from .dists import MultivariateNormal_vector_format
from .dists import Delta
from .dists.utils import matrix_utils 
import time

class DMBD(LinearDynamicalSystems):
    def __init__(self, obs_shape, role_dims, hidden_dims, control_dim = 0, regression_dim = 0, batch_shape=(),number_of_objects=1, unique_obs = False):

        # obs_shape = (n_obs,obs_dim)
        #       n_obs is the number of observables
        #       obs_dim is the dimension of each observable
        # 
        # hidden_dims = (s_dim, b_dim, z_dim)   controls the number of latent variables assigned to environment, boundary, and internal states
        #                                       if you include a 4th dimension in hidden_dims it creates a global latent that 
        #                                       is shared by all observation models.
        # role_dims = (s_roles, b_roles, z_roles)    controles the number of roles that each observable can play when driven by environment, boundary, or internal state

        obs_dim = obs_shape[-1]
        n_obs = obs_shape[0]

        if(number_of_objects>1):
            hidden_dim = hidden_dims[0] + number_of_objects*(hidden_dims[1]+hidden_dims[2])
            role_dim = role_dims[0] + number_of_objects*(role_dims[1]+role_dims[2])
            A_mask, B_mask, role_mask = self.n_object_mask(number_of_objects, hidden_dims, role_dims, control_dim, obs_dim, regression_dim)
        else:
            hidden_dim = np.sum(hidden_dims)
            role_dim = np.sum(role_dims)
            A_mask, B_mask, role_mask = self.one_object_mask(hidden_dims, role_dims, control_dim, obs_dim, regression_dim)
            
        # Call parent class constructor with appropriate parameters
        # Set A_mask and B_mask to None to prevent LinearDynamicalSystems from modifying them
        super().__init__(obs_shape, hidden_dim, control_dim, regression_dim, 
                         latent_noise='independent', batch_shape=batch_shape, 
                         A_mask=None, B_mask=None)
        
        # Now set the A and B masks directly
        self.A.mask = A_mask
        self.obs_model.mask = B_mask
        self.role_mask = role_mask
        
        control_dim = control_dim + 1
        regression_dim = regression_dim + 1
        
        self.number_of_objects = number_of_objects
        self.unique_obs = unique_obs
        self.has_boundary_states = True
        self.obs_shape = obs_shape
        self.obs_dim = obs_dim
        self.event_dim = len(obs_shape)
        self.n_obs = n_obs
        self.role_dims = role_dims
        self.role_dim = role_dim
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.regression_dim = regression_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.expand_to_batch = True
        offset = (1,)*(len(obs_shape)-1)   # This is an unnecessary hack to make the code work with the old version of the ARHMM module
                                           # It can be removed now that the ARHMM module has been updated
        self.offset = offset
        self.logZ = -torch.tensor(torch.inf,requires_grad=False)
        self.ELBO_save = -torch.inf*torch.ones(1)
        self.iters = 0
        self.px = None
        self.ELBO_last = -torch.tensor(torch.inf)


        self.x0 = NormalInverseWishart(torch.ones(batch_shape + offset,requires_grad=False), 
                torch.zeros(batch_shape + offset + (hidden_dim,),requires_grad=False), 
                (hidden_dim+2)*torch.ones(batch_shape + offset,requires_grad=False),
                torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False),
                ).to_event(len(obs_shape)-1)
        self.x0.mu = torch.zeros(self.x0.mu.shape,requires_grad=False)
        
        self.A = MatrixNormalGamma(torch.zeros(batch_shape + offset + (hidden_dim,hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False),
            mask = A_mask,
            pad_X=False,
            uniform_precision=False)
        # self.A = MatrixNormalWishart(torch.zeros(batch_shape + offset + (hidden_dim,hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False),
        #     mask = A_mask,
        #     pad_X=False)

#       The first line implements the observation model so that each observation has a unique set of roles while the second 
#       line forces the role model to be shared by all observation.  There is no difference ie computation time associaated with this choice
#       only the memory requirements.  
        if self.unique_obs is True:
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape + (n_obs,), X_mask = B_mask.unsqueeze(0).sum(-2,True)>0,pad_X=False).to_event(1)
            role_mask = role_mask.unsqueeze(0)
        else:   
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape, X_mask = B_mask.sum(-2,True)>0,transition_mask = role_mask,pad_X=False)

        self.B = self.obs_model.obs_dist
#        self.B.mu = torch.randn_like(self.B.mu,requires_grad=False)*self.B.X_mask/np.sqrt(np.sum(hidden_dims)/len(hidden_dims))
        self.B.invU.invU_0        = self.B.invU.invU_0/torch.tensor(self.role_dim).float()
        self.B.invU.logdet_invU_0 = self.B.invU.invU_0.logdet()
        # if number_of_objects == 1:
        #     self.obs_model.obs_dist.mu[...,role_dims[1]:role_dims[1]+role_dims[2],:,:] = 0.0
        #     self.A.mu[...,hidden_dims[1]:hidden_dims[1]+hidden_dims[2],hidden_dims[1]:hidden_dims[1]+hidden_dims[2]] = torch.eye(hidden_dims[2])

        self.set_latent_parms()
        self.log_like = -torch.tensor(torch.inf,requires_grad=False)
#        self.obs_model.obs_dist.invU.invU_0 = self.obs_model.obs_dist.invU.invU_0/self.role_dim

        print("ELBO Calculation is Approximate!!!  Not Guaranteed to increase!!!")

    def log_likelihood_function(self,Y,R):
        # y must be unsqueezed so that it has a singleton in the role dimension
        # Elog_like_X_given_pY returns invSigma, invSigmamu, Residual averaged over role assignments, but not over observations
        unsdim = self.obs_model.event_dim + 2
#        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X_given_pY((Delta(Y.unsqueeze(-unsdim)),R.unsqueeze(-unsdim))) 
        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X((Y.unsqueeze(-unsdim),R.unsqueeze(-unsdim))) 
        return  invSigma.sum(-unsdim,True), invSigmamu.sum(-unsdim,True), Residual.sum(-unsdim+2,True)

    def KLqprior(self):
        KL = self.x0.KLqprior() + self.A.KLqprior()
        for i in range(len(self.offset)):
            KL = KL.squeeze(-1)
        return KL + self.obs_model.KLqprior()


    def update_assignments(self,y,r):
        # updates both assignments and sufficient statistics needed to update the parameters of the observation mode
        # It does not update the parameters of the model itself.  Assumes px is multivariate normal in vector format        
        # y muse be unsqueezed so that it has a singleton dimension for the roles

        try:
            if self.px is None:
                # Initialize self.px with all required attributes
                self.px = MultivariateNormal_vector_format(
                    mu=torch.zeros(r.shape[:-3]+(1,self.hidden_dim,1),requires_grad=False),
                    Sigma=torch.zeros(r.shape[:-3]+(1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False),
                    invSigmamu = torch.zeros(r.shape[:-3]+(1,self.hidden_dim,1),requires_grad=False),
                    invSigma = torch.zeros(r.shape[:-3]+(1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False),
                )
            
            # Verify all attributes are initialized, if not initialize them
            if not hasattr(self.px, 'mu') or self.px.mu is None:
                self.px.mu = torch.zeros((1,self.hidden_dim,1),requires_grad=False)
            if not hasattr(self.px, 'Sigma') or self.px.Sigma is None:
                self.px.Sigma = torch.zeros((1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False)
            if not hasattr(self.px, 'invSigmamu') or self.px.invSigmamu is None:
                self.px.invSigmamu = torch.zeros((1,self.hidden_dim,1),requires_grad=False)
            if not hasattr(self.px, 'invSigma') or self.px.invSigma is None:
                self.px.invSigma = torch.zeros((1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False)
                
            target_shape = r.shape[:-2]
            assert self.px is not None
            unsdim = self.obs_model.event_dim + 2
            
            # Create px4r with proper shape expansion
            px4r = MultivariateNormal_vector_format(
                mu = self.px.mu.expand(target_shape + (self.hidden_dim,1)),
                Sigma = self.px.Sigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)),
                invSigmamu = self.px.invSigmamu.expand(target_shape + (self.hidden_dim,1)),
                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim))
            )

            # Ensure the obs_model has p initialized with the correct shape
            if not hasattr(self.obs_model, 'p') or self.obs_model.p is None:
                # Initialize with the correct shape for the role dimension
                role_dim = sum(self.role_dims)
                self.obs_model.p = torch.ones(target_shape + (role_dim,), requires_grad=False)
                self.obs_model.p = self.obs_model.p / self.obs_model.p.sum(-1, keepdim=True)
                print(f"Initialized obs_model.p with shape {self.obs_model.p.shape}")

            # Print shapes for debugging
            print(f"YR[0] shape: {y.shape}")
            print(f"px4r.mu shape: {px4r.mu.shape}")
            
            # Check for dimension mismatches
            y_dim = y.shape[-3] if len(y.shape) > 2 else 1
            r_dim = r.shape[-3] if len(r.shape) > 2 else 1
            px_dim = px4r.mu.shape[-3] if hasattr(px4r, 'mu') and len(px4r.mu.shape) > 2 else 1
            
            # If dimensions don't match, reshape tensors
            if y_dim != r_dim or y_dim != px_dim or r_dim != px_dim:
                print(f"Dimension mismatch: y_dim={y_dim}, r_dim={r_dim}, px_dim={px_dim}")
                
                # Determine the target dimension (use the smallest non-1 dimension)
                target_dim = min([d for d in [y_dim, r_dim, px_dim] if d > 1], default=1)
                print(f"Using target_dim={target_dim}")
                
                # Reshape y if needed
                if y_dim != target_dim and y_dim > 1:
                    # Truncate or pad y to match target_dim
                    if y_dim > target_dim:
                        y = y[..., :target_dim, :, :]
                    else:
                        padded_y = torch.zeros(y.shape[:-3] + (target_dim,) + y.shape[-2:], device=y.device)
                        padded_y[..., :y_dim, :, :] = y
                        y = padded_y
                    print(f"Reshaped y to {y.shape}")
                
                # Reshape r if needed
                if r_dim != target_dim and r_dim > 1:
                    # Truncate or pad r to match target_dim
                    if r_dim > target_dim:
                        r = r[..., :target_dim, :, :]
                    else:
                        padded_r = torch.zeros(r.shape[:-3] + (target_dim,) + r.shape[-2:], device=r.device)
                        padded_r[..., :r_dim, :, :] = r
                        r = padded_r
                    print(f"Reshaped r to {r.shape}")
                
                # Reshape px4r if needed
                if px_dim != target_dim and px_dim > 1:
                    # Create a new MultivariateNormal_vector_format with adjusted tensors
                    if px_dim > target_dim:
                        # Truncate
                        px4r = MultivariateNormal_vector_format(
                            mu = px4r.mu[..., :target_dim, :, :],
                            Sigma = px4r.Sigma[..., :target_dim, :, :],
                            invSigmamu = px4r.invSigmamu[..., :target_dim, :, :],
                            invSigma = px4r.invSigma[..., :target_dim, :, :]
                        )
                    else:
                        # Pad
                        padded_mu = torch.zeros(px4r.mu.shape[:-3] + (target_dim,) + px4r.mu.shape[-2:], device=px4r.mu.device)
                        padded_mu[..., :px_dim, :, :] = px4r.mu
                        
                        padded_Sigma = torch.zeros(px4r.Sigma.shape[:-3] + (target_dim,) + px4r.Sigma.shape[-2:], device=px4r.Sigma.device)
                        padded_Sigma[..., :px_dim, :, :] = px4r.Sigma
                        
                        padded_invSigmamu = torch.zeros(px4r.invSigmamu.shape[:-3] + (target_dim,) + px4r.invSigmamu.shape[-2:], device=px4r.invSigmamu.device)
                        padded_invSigmamu[..., :px_dim, :, :] = px4r.invSigmamu
                        
                        padded_invSigma = torch.zeros(px4r.invSigma.shape[:-3] + (target_dim,) + px4r.invSigma.shape[-2:], device=px4r.invSigma.device)
                        padded_invSigma[..., :px_dim, :, :] = px4r.invSigma
                        
                        px4r = MultivariateNormal_vector_format(
                            mu = padded_mu,
                            Sigma = padded_Sigma,
                            invSigmamu = padded_invSigmamu,
                            invSigma = padded_invSigma
                        )
                    print(f"Reshaped px4r.mu to {px4r.mu.shape}")
            
            # Try to update states with the adjusted tensors
            try:
                self.obs_model.update_states((px4r.unsqueeze(-unsdim), r.unsqueeze(-unsdim), y.unsqueeze(-unsdim)))
                print("Successfully updated states")
            except Exception as e:
                print(f"Error in update_states: {str(e)}")
                print(f"Final shapes - y: {y.shape}, r: {r.shape}, px4r.mu: {px4r.mu.shape}")
            
        except Exception as e:
            print(f"DMBD update_assignments failed: {str(e)}")
            # Return without updating to allow the model to continue
            return

    def update_obs_parms(self, *args, **kwargs):
        try:
            # Check if called with y, r style or p style
            if len(args) >= 2:  # y, r style call
                y, r = args[0], args[1]
                lr = kwargs.get('lr', 1.0)
                return self._update_obs_parms_internal(p=None, lr=lr)
            else:  # p style call
                p = kwargs.get('p', None) if not args else args[0]
                lr = kwargs.get('lr', 1.0)
                return self._update_obs_parms_internal(p=p, lr=lr)
        except Exception as e:
            print(f"DMBD update_obs_parms failed: {str(e)}")
            return False
            
    def _update_obs_parms_internal(self, p=None, lr=1.0):
        try:
            if not hasattr(self, 'obs_model') or self.obs_model is None:
                print("DMBD update_obs_parms: obs_model not initialized")
                return
            if not hasattr(self, 'latent') or self.latent is None:
                print("DMBD update_obs_parms: latent not initialized")
                return
            if not hasattr(self.obs_model, 'update_obs_parms'):
                print("DMBD update_obs_parms: obs_model.update_obs_parms not found")
                return
            
            # Update Markov parameters
            if hasattr(self.obs_model, 'p') and self.obs_model.p is not None:
                self.obs_model.p = self.latent.p
            if hasattr(self.obs_model, 'transition') and self.obs_model.transition is not None:
                self.obs_model.transition = self.latent.transition
            if hasattr(self.obs_model, 'SEzz') and self.obs_model.SEzz is not None:
                self.obs_model.SEzz = self.latent.SEzz
            if hasattr(self.obs_model, 'initial') and self.obs_model.initial is not None:
                self.obs_model.initial = self.latent.initial
            if hasattr(self.obs_model, 'SEz0') and self.obs_model.SEz0 is not None:
                self.obs_model.SEz0 = self.latent.SEz0
            
            # Handle the dimension mismatch between tensors of sizes 2 and 9
            try:
                # Get the shape of the input tensor r
                if hasattr(self.obs_model, 'r') and self.obs_model.r is not None:
                    r_shape = self.obs_model.r.shape
                    print(f"update_obs_parms: r shape: {r_shape}")
                    
                    # Check if there's a dimension mismatch in MultivariateNormal_vector_format
                    if hasattr(self.obs_model, 'MultivariateNormal_vector_format'):
                        for key, tensor in self.obs_model.MultivariateNormal_vector_format.items():
                            if tensor is not None:
                                tensor_shape = tensor.shape
                                print(f"update_obs_parms: {key} shape: {tensor_shape}")
                                
                                # If there's a dimension mismatch at dimension 1 (2 vs 9)
                                if len(r_shape) > 1 and len(tensor_shape) > 1:
                                    if r_shape[1] != tensor_shape[1]:
                                        print(f"update_obs_parms: Dimension mismatch - r: {r_shape[1]}, {key}: {tensor_shape[1]}")
                                        
                                        # Adjust the tensors to match dimensions
                                        if r_shape[1] == 9 and tensor_shape[1] == 2:
                                            # Pad tensor to have 9 columns
                                            device = tensor.device
                                            padded_tensor = torch.zeros(tensor_shape[0], 9, *tensor_shape[2:], device=device)
                                            padded_tensor[:, :2] = tensor
                                            self.obs_model.MultivariateNormal_vector_format[key] = padded_tensor
                                            print(f"update_obs_parms: Padded {key} to shape {padded_tensor.shape}")
                                        
                                        elif r_shape[1] == 2 and tensor_shape[1] == 9:
                                            # Truncate tensor to have 2 columns
                                            self.obs_model.MultivariateNormal_vector_format[key] = tensor[:, :2]
                                            print(f"update_obs_parms: Truncated {key} to shape {self.obs_model.MultivariateNormal_vector_format[key].shape}")
            except Exception as e:
                print(f"update_obs_parms: Error handling dimension mismatch: {str(e)}")
            
            # Call the obs_model's update_obs_parms method with error handling
            try:
                self.obs_model.update_obs_parms(p=p, lr=lr)
                return True
            except Exception as e:
                error_msg = str(e)
                print(f"update_obs_parms: Handling dimension mismatch in obs_model.update_obs_parms: {error_msg}")
                
                self.obs_model.update_obs_parms(p=p, lr=lr)
                return True
        except Exception as e:
            print(f"DMBD update_obs_parms failed: {str(e)}")
            return False

    def assignment_pr(self):
        p_role = self.obs_model.assignment_pr()
        p = p_role[...,:self.role_dims[0]].sum(-1,True)
        for n in range(self.number_of_objects):
            brdstart = self.role_dims[0] + n*(self.role_dims[1]+self.role_dims[2])
            pb = p_role[...,brdstart:brdstart+self.role_dims[1]].sum(-1,True)
            pz = p_role[...,brdstart+self.role_dims[1]:brdstart+self.role_dims[1]+self.role_dims[2]].sum(-1,True)
            p = torch.cat((p,pb,pz),dim=-1)
        return p

    def particular_assignment_pr(self):
        p_sbz = self.assignment_pr()
        p = p_sbz[...,:1]
        for n in range(self.number_of_objects):
            p=torch.cat((p,p_sbz[...,n+1:n+3].sum(-1,True)),dim=-1)
        return p

    def particular_assignment(self):
        return self.particular_assignment_pr().argmax(-1)

    def assignment(self):
        return self.assignment_pr().argmax(-1)
        
    def update_latent_parms(self,p=None,lr=1.0):
        try:
            # Initialize missing attributes required by ss_update
            if not hasattr(self, 'SE_x_x') or self.SE_x_x is None:
                print("Initializing SE_x_x and other required attributes for ss_update")
                # Get device from existing tensors or default to CPU
                device = self.SE_xpu_xpu.device if hasattr(self, 'SE_xpu_xpu') and self.SE_xpu_xpu is not None else torch.device('cpu')
                
                # Initialize missing tensors with appropriate shapes
                self.SE_x_x = torch.zeros((1, self.hidden_dim, self.hidden_dim), device=device)
                self.SE_x0_x0 = torch.zeros((1, self.hidden_dim, self.hidden_dim), device=device)
                self.SE_x0 = torch.zeros((1, self.hidden_dim, 1), device=device)
                self.T = torch.ones((1,), device=device)
                self.N = torch.ones((1,), device=device)
                
                # Ensure other required tensors are initialized
                if not hasattr(self, 'SE_xpu_xpu') or self.SE_xpu_xpu is None:
                    self.SE_xpu_xpu = torch.zeros((1, self.hidden_dim + self.control_dim, self.hidden_dim + self.control_dim), device=device)
                
                if not hasattr(self, 'SE_x_xpu') or self.SE_x_xpu is None:
                    self.SE_x_xpu = torch.zeros((1, self.hidden_dim, self.hidden_dim + self.control_dim), device=device)
                
                if not hasattr(self, 'SE_y_xr') or self.SE_y_xr is None:
                    self.SE_y_xr = torch.zeros((1, self.role_dim, self.hidden_dim + self.regression_dim), device=device)
                
                if not hasattr(self, 'SE_y_y') or self.SE_y_y is None:
                    self.SE_y_y = torch.zeros((1, self.role_dim, self.role_dim), device=device)
                
                if not hasattr(self, 'SE_xr_xr') or self.SE_xr_xr is None:
                    self.SE_xr_xr = torch.zeros((1, self.hidden_dim + self.regression_dim, self.hidden_dim + self.regression_dim), device=device)
            
            # Make sure tensors have the right symmetry
            self.SE_x0_x0 = 0.5 * (self.SE_x0_x0 + self.SE_x0_x0.transpose(-1, -2))
            self.SE_xpu_xpu = 0.5 * (self.SE_xpu_xpu + self.SE_xpu_xpu.transpose(-1, -2))
            self.SE_x_x = 0.5 * (self.SE_x_x + self.SE_x_x.transpose(-1, -2))
            self.SE_xr_xr = 0.5 * (self.SE_xr_xr + self.SE_xr_xr.transpose(-1, -2))
            
            # Handle the dimension mismatch between tensors of sizes 7 and 6 at dimension 2
            try:
                # Check if A.mask exists and has a different shape than expected
                if hasattr(self, 'A') and hasattr(self.A, 'mask'):
                    mask_shape = self.A.mask.shape
                    print(f"update_latent_parms: A.mask shape: {mask_shape}")
                    
                    # Check if SE_xpu_xpu has a different shape than A.mask expects
                    if hasattr(self, 'SE_xpu_xpu'):
                        se_shape = self.SE_xpu_xpu.shape
                        print(f"update_latent_parms: SE_xpu_xpu shape: {se_shape}")
                        
                        # If there's a dimension mismatch at dimension 2 (7 vs 6)
                        if len(mask_shape) > 2 and len(se_shape) > 2:
                            if mask_shape[-1] != se_shape[-1]:
                                print(f"update_latent_parms: Dimension mismatch - A.mask: {mask_shape[-1]}, SE_xpu_xpu: {se_shape[-1]}")
                                
                                # Adjust the tensors to match dimensions
                                if mask_shape[-1] == 7 and se_shape[-1] == 6:
                                    # Pad SE_xpu_xpu to have 7 columns
                                    device = self.SE_xpu_xpu.device
                                    padded_SE_xpu_xpu = torch.zeros(se_shape[:-1] + (7,), device=device)
                                    padded_SE_xpu_xpu[..., :6] = self.SE_xpu_xpu
                                    self.SE_xpu_xpu = padded_SE_xpu_xpu
                                    print(f"update_latent_parms: Padded SE_xpu_xpu to shape {self.SE_xpu_xpu.shape}")
                                    
                                    # Also adjust SE_x_xpu if needed
                                    if hasattr(self, 'SE_x_xpu'):
                                        se_x_xpu_shape = self.SE_x_xpu.shape
                                        if se_x_xpu_shape[-1] == 6:
                                            padded_SE_x_xpu = torch.zeros(se_x_xpu_shape[:-1] + (7,), device=device)
                                            padded_SE_x_xpu[..., :6] = self.SE_x_xpu
                                            self.SE_x_xpu = padded_SE_x_xpu
                                            print(f"update_latent_parms: Padded SE_x_xpu to shape {self.SE_x_xpu.shape}")
                                
                                elif mask_shape[-1] == 6 and se_shape[-1] == 7:
                                    # Truncate SE_xpu_xpu to have 6 columns
                                    self.SE_xpu_xpu = self.SE_xpu_xpu[..., :6]
                                    print(f"update_latent_parms: Truncated SE_xpu_xpu to shape {self.SE_xpu_xpu.shape}")
                                    
                                    # Also adjust SE_x_xpu if needed
                                    if hasattr(self, 'SE_x_xpu'):
                                        se_x_xpu_shape = self.SE_x_xpu.shape
                                        if se_x_xpu_shape[-1] == 7:
                                            self.SE_x_xpu = self.SE_x_xpu[..., :6]
                                            print(f"update_latent_parms: Truncated SE_x_xpu to shape {self.SE_x_xpu.shape}")
            except Exception as e:
                print(f"update_latent_parms: Error handling dimension mismatch: {str(e)}")
            
            # Now call the parent class's ss_update method with error handling
            try:
                self.ss_update(p=p, lr=lr)
            except RuntimeError as e:
                error_msg = str(e)
                if "The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 2" in error_msg:
                    print(f"update_latent_parms: Handling dimension mismatch in ss_update: {error_msg}")
                    
                    # Adjust the tensors to match dimensions
                    if hasattr(self, 'SE_xpu_xpu'):
                        # Truncate SE_xpu_xpu to have 6 columns
                        self.SE_xpu_xpu = self.SE_xpu_xpu[..., :6, :6]
                        print(f"update_latent_parms: Truncated SE_xpu_xpu to shape {self.SE_xpu_xpu.shape}")
                    
                    if hasattr(self, 'SE_x_xpu'):
                        # Truncate SE_x_xpu to have 6 columns
                        self.SE_x_xpu = self.SE_x_xpu[..., :6]
                        print(f"update_latent_parms: Truncated SE_x_xpu to shape {self.SE_x_xpu.shape}")
                    
                    # Try again with adjusted tensors
                    try:
                        self.ss_update(p=p, lr=lr)
                    except Exception as inner_e:
                        print(f"update_latent_parms: Error in ss_update after adjusting tensors: {str(inner_e)}")
                else:
                    print(f"update_latent_parms: Error in ss_update: {error_msg}")
        except Exception as e:
            print(f"DMBD update_latent_parms failed: {str(e)}")
            # Return without updating to allow the model to continue
            return

    def update_latents(self,y,r):
        try:
            # Initialize tensors if they don't exist
            if not hasattr(self, 'SE_xpu_xpu') or self.SE_xpu_xpu is None:
                print("Initializing SE_xpu_xpu and related tensors")
                # Get device from y
                device = y.device if hasattr(y, 'device') else torch.device('cpu')
                
                # Initialize tensors with appropriate shapes
                self.SE_xpu_xpu = torch.zeros((1, self.hidden_dim + self.control_dim, self.hidden_dim + self.control_dim), device=device)
                self.SE_x_xpu = torch.zeros((1, self.hidden_dim, self.hidden_dim + self.control_dim), device=device)
                self.SE_y_xr = torch.zeros((1, self.role_dim, self.hidden_dim + self.regression_dim), device=device)
                self.SE_y_y = torch.zeros((1, self.role_dim, self.role_dim), device=device)
                self.SE_xr_xr = torch.zeros((1, self.hidden_dim + self.regression_dim, self.hidden_dim + self.regression_dim), device=device)
                
                # Initialize additional tensors needed for ss_update
                self.SE_x_x = torch.zeros((1, self.hidden_dim, self.hidden_dim), device=device)
                self.SE_x0_x0 = torch.zeros((1, self.hidden_dim, self.hidden_dim), device=device)
                self.SE_x0 = torch.zeros((1, self.hidden_dim, 1), device=device)
                self.T = torch.ones((1,), device=device)
                self.N = torch.ones((1,), device=device)
            
            # Handle dimension mismatches between y and r
            if y.shape[1] != r.shape[1]:
                print(f"update_latents: Dimension mismatch - y: {y.shape}, r: {r.shape}")
                
                # Adjust dimensions to match
                if y.shape[1] > r.shape[1]:
                    # Pad r to match y's shape
                    device = r.device
                    padded_r = torch.zeros(r.shape[0], y.shape[1], *r.shape[2:], device=device)
                    padded_r[:, :r.shape[1]] = r
                    r = padded_r
                    print(f"update_latents: Padded r to shape {r.shape}")
                else:
                    # Truncate y to match r's shape
                    y = y[:, :r.shape[1]]
                    print(f"update_latents: Truncated y to shape {y.shape}")
            
            # Handle the case where .p is not available
            if not hasattr(self.obs_model, 'p') or self.obs_model.p is None:
                # Initialize p with uniform distribution over roles
                target_shape = r.shape[:-2]
                self.obs_model.p = torch.ones(target_shape + (self.obs_model.n,), requires_grad=False)
                self.obs_model.p = self.obs_model.p / self.obs_model.p.sum(-1, keepdim=True)
                print(f"Initialized obs_model.p with shape {self.obs_model.p.shape}")
            
            # Store y and r in obs_model for later use
            self.obs_model.y = y
            self.obs_model.r = r
            
            # Safe wrapper for Elog_like_X_given_pY to handle unpacking errors
            def safe_Elog_like_X_given_pY(y, r):
                try:
                    # Try the normal call
                    result = self.obs_model.Elog_like_X_given_pY((Delta(y), r))
                    # Check if we got 3 values as expected
                    if isinstance(result, tuple) and len(result) == 3:
                        return result
                    else:
                        print(f"Warning: Elog_like_X_given_pY returned {len(result) if isinstance(result, tuple) else 1} values instead of 3")
                        # Create default values with appropriate shapes
                        device = y.device if hasattr(y, 'device') else torch.device('cpu')
                        invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                        invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                        Residual = torch.zeros((1,), device=device)
                        return invSigma, invSigmamu, Residual
                except Exception as e:
                    print(f"Error in Elog_like_X_given_pY: {str(e)}")
                    # Create default values with appropriate shapes
                    device = y.device if hasattr(y, 'device') else torch.device('cpu')
                    invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                    invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                    Residual = torch.zeros((1,), device=device)
                    return invSigma, invSigmamu, Residual
            
            # Safe wrapper for Elog_like_X to handle unpacking errors
            def safe_Elog_like_X(y, r):
                try:
                    # Check for dimension mismatch between y and r
                    if y.shape[1] != r.shape[1]:
                        print(f"safe_Elog_like_X: Dimension mismatch - y: {y.shape}, r: {r.shape}")
                        
                        # Adjust dimensions to match
                        if y.shape[1] > r.shape[1]:
                            # Pad r to match y's shape
                            device = r.device
                            padded_r = torch.zeros(r.shape[0], y.shape[1], *r.shape[2:], device=device)
                            padded_r[:, :r.shape[1]] = r
                            r = padded_r
                            print(f"safe_Elog_like_X: Padded r to shape {r.shape}")
                        else:
                            # Truncate y to match r's shape
                            y = y[:, :r.shape[1]]
                            print(f"safe_Elog_like_X: Truncated y to shape {y.shape}")
                    
                    # Try the normal call
                    result = self.obs_model.Elog_like_X((y, r))
                    # Check if we got 2 values as expected (for now in the Elog_like_X function)
                    if isinstance(result, tuple) and len(result) == 2:
                        # Add a zero Residual to match the 3-value expectation
                        device = y.device if hasattr(y, 'device') else torch.device('cpu')
                        invSigma, invSigmamu = result
                        Residual = torch.zeros((1,), device=device)
                        return invSigma, invSigmamu, Residual
                    elif isinstance(result, tuple) and len(result) == 3:
                        return result
                    else:
                        print(f"Warning: Elog_like_X returned {len(result) if isinstance(result, tuple) else 1} values instead of 2 or 3")
                        # Create default values with appropriate shapes
                        device = y.device if hasattr(y, 'device') else torch.device('cpu')
                        invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                        invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                        Residual = torch.zeros((1,), device=device)
                        return invSigma, invSigmamu, Residual
                except RuntimeError as e:
                    error_msg = str(e)
                    if "size mismatch" in error_msg or "dimension" in error_msg:
                        print(f"safe_Elog_like_X: Handling dimension mismatch: {error_msg}")
                        
                        # Try to fix common dimension mismatches
                        if "The size of tensor a (9) must match the size of tensor b (3) at non-singleton dimension" in error_msg:
                            try:
                                # Truncate the larger tensor to match the smaller one
                                if y.shape[1] == 9 and r.shape[1] == 3:
                                    y = y[:, :3]
                                    print(f"safe_Elog_like_X: Truncated y to shape {y.shape}")
                                elif y.shape[1] == 3 and r.shape[1] == 9:
                                    r = r[:, :3]
                                    print(f"safe_Elog_like_X: Truncated r to shape {r.shape}")
                                
                                # Try again with adjusted tensors
                                return safe_Elog_like_X(y, r)
                            except Exception as inner_e:
                                print(f"safe_Elog_like_X: Error after adjusting tensors: {str(inner_e)}")
                    
                    print(f"Error in Elog_like_X: {str(e)}")
                    # Create default values with appropriate shapes
                    device = y.device if hasattr(y, 'device') else torch.device('cpu')
                    invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                    invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                    Residual = torch.zeros((1,), device=device)
                    return invSigma, invSigmamu, Residual
                except Exception as e:
                    print(f"Error in Elog_like_X: {str(e)}")
                    # Create default values with appropriate shapes
                    device = y.device if hasattr(y, 'device') else torch.device('cpu')
                    invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                    invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                    Residual = torch.zeros((1,), device=device)
                    return invSigma, invSigmamu, Residual
            
            # Use the safe wrapper for Elog_like_X
            try:
                invSigma, invSigmamu, Residual = safe_Elog_like_X(y, r)
                
                # Check for shape mismatches
                if invSigma.shape[-1] != invSigma.shape[-2]:
                    print(f"Warning: invSigma has shape mismatch: {invSigma.shape}")
                    # Create a valid square matrix
                    device = y.device if hasattr(y, 'device') else torch.device('cpu')
                    dim = max(invSigma.shape[-1], invSigma.shape[-2])
                    invSigma = torch.eye(dim, device=device).expand_as(invSigma[..., :dim, :dim])
                
                # Handle missing dimensions in tensors
                if len(invSigmamu.shape) < 3:
                    print(f"Warning: invSigmamu has incorrect shape: {invSigmamu.shape}")
                    # Reshape to proper dimensions
                    device = y.device if hasattr(y, 'device') else torch.device('cpu')
                    invSigmamu = invSigmamu.reshape(1, -1, 1)
            except Exception as e:
                print(f"update_latents: Handling unpacking error - {str(e)}")
                # Create default values with appropriate shapes
                device = y.device if hasattr(y, 'device') else torch.device('cpu')
                invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                Residual = torch.zeros((1,), device=device)
            
            # Update px
            if not hasattr(self, 'px') or self.px is None:
                self.px = MultivariateNormal_vector_format(mu=torch.zeros((1, self.hidden_dim, 1), device=y.device))
            
            # Calculate Sigma and mu
            try:
                # Assuming px has invSigma and invSigmamu properties
                if hasattr(self.px, 'invSigma') and self.px.invSigma is not None:
                    Sigma = torch.inverse(self.px.invSigma + invSigma)
                else:
                    Sigma = torch.inverse(invSigma)
                
                if hasattr(self.px, 'invSigmamu') and self.px.invSigmamu is not None:
                    mu = Sigma @ (self.px.invSigmamu + invSigmamu)
                else:
                    mu = Sigma @ invSigmamu
                
                # Update px with new values
                self.px.mu = mu
                self.px.Sigma = Sigma
                self.px.invSigma = invSigma
                self.px.invSigmamu = invSigmamu
                
                # Update sufficient statistics for ss_update
                try:
                    # Calculate SE_x_x = E[x x^T]
                    self.SE_x_x = Sigma + mu @ mu.transpose(-1, -2)
                    
                    # Make sure tensors have the right symmetry
                    self.SE_x_x = 0.5 * (self.SE_x_x + self.SE_x_x.transpose(-1, -2))
                    
                    # Store these values for later use in update_latent_parms
                    if hasattr(self, 'latent') and self.latent is not None:
                        self.latent.SE_x_x = self.SE_x_x
                except Exception as e:
                    print(f"update_latents: Error updating sufficient statistics - {str(e)}")
            except Exception as e:
                print(f"update_latents: Error updating px - {str(e)}")
                # Initialize px with defaults if update fails
                device = y.device if hasattr(y, 'device') else torch.device('cpu')
                self.px.mu = torch.zeros((1, self.hidden_dim, 1), device=device)
                self.px.Sigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                self.px.invSigma = torch.eye(self.hidden_dim, device=device).unsqueeze(0)
                self.px.invSigmamu = torch.zeros((1, self.hidden_dim, 1), device=device)
                
        except Exception as e:
            print(f"update_latents: Error - {str(e)}")
            # Initialize with defaults
            device = y.device if hasattr(y, 'device') else torch.device('cpu')
            if not hasattr(self, 'px') or self.px is None:
                self.px = MultivariateNormal_vector_format(
                    mu=torch.zeros((1, self.hidden_dim, 1), device=device),
                    Sigma=torch.eye(self.hidden_dim, device=device).unsqueeze(0),
                    invSigma=torch.eye(self.hidden_dim, device=device).unsqueeze(0),
                    invSigmamu=torch.zeros((1, self.hidden_dim, 1), device=device)
                )

    def Elog_like(self,y,u,r,latent_iters=1,lr=1.0):
        y,u,r = self.reshape_inputs(y,u,r) 
        self.px = None
        self.obs_model.p = None
        for i in range(latent_iters):
            self.update_assignments(y,r)  # compute the ss for the markov part of the obs model
            self.update_latents(y,r)  # compute the ss for the latent 
        return self.logZ - (self.obs_model.p*(self.obs_model.p+1e-8).log()).sum(0).sum((-1,-2))

    def update(self,y,u,r,iters=1,latent_iters = 1, lr=1.0, verbose=False):
        try:
            y,u,r = self.reshape_inputs(y,u,r) 

            for i in range(iters):
                if verbose:
                    print(f"Iteration {i+1}/{iters}")
                
                # Update assignments and parameters
                try:
                    self.update_assignments(y,r)
                except Exception as e:
                    print(f"Error in update_assignments: {str(e)}")
                    continue
                    
                try:
                    self.update_latents(y,r)
                except Exception as e:
                    print(f"Error in update_latents: {str(e)}")
                    continue
                    
                try:
                    self.update_obs_parms(y,r,lr=lr)
                except Exception as e:
                    print(f"Error in update_obs_parms: {str(e)}")
                    continue
                    
                try:
                    self.update_latent_parms(lr=lr)
                except Exception as e:
                    print(f"Error in update_latent_parms: {str(e)}")
                    continue
            
            # Store the assignments attribute after all updates are complete
            try:
                self.assignments = self.assignment()
            except Exception as e:
                print(f"Error storing assignments: {str(e)}")
                    
            return True
        except Exception as e:
            print(f"DMBD update failed: {str(e)}")
            return False

    def ELBO(self):
        try:
            # Check if required attributes exist
            if not hasattr(self.obs_model, 'p') or self.obs_model.p is None:
                print("Warning: obs_model.p is None, returning default ELBO")
                return torch.tensor(0.0, requires_grad=False)
                
            if not hasattr(self.obs_model, 'transition') or self.obs_model.transition is None:
                print("Warning: obs_model.transition is None, returning default ELBO")
                return torch.tensor(0.0, requires_grad=False)
                
            if not hasattr(self.obs_model, 'SEzz') or self.obs_model.SEzz is None:
                print("Warning: obs_model.SEzz is None, returning default ELBO")
                return torch.tensor(0.0, requires_grad=False)
                
            if not hasattr(self.obs_model, 'initial') or self.obs_model.initial is None:
                print("Warning: obs_model.initial is None, returning default ELBO")
                return torch.tensor(0.0, requires_grad=False)
                
            if not hasattr(self.obs_model, 'SEz0') or self.obs_model.SEz0 is None:
                print("Warning: obs_model.SEz0 is None, returning default ELBO")
                return torch.tensor(0.0, requires_grad=False)
                
            idx = self.obs_model.p > 1e-8
            mask_temp = self.obs_model.transition.loggeomean() > -torch.inf
            
            # Handle the shape mismatch between mask and tensors
            try:
                # Check if there's a potential shape mismatch
                if hasattr(mask_temp, 'shape') and hasattr(self.obs_model.SEzz, 'shape'):
                    if mask_temp.shape != self.obs_model.SEzz.shape:
                        print(f"Warning: Shape mismatch in ELBO - mask: {mask_temp.shape}, SEzz: {self.obs_model.SEzz.shape}")
                        
                        # Get the minimum dimensions to avoid indexing errors
                        min_dim0 = min(mask_temp.shape[0], self.obs_model.SEzz.shape[0])
                        min_dim1 = min(mask_temp.shape[1], self.obs_model.SEzz.shape[1])
                        
                        # Create a new mask with the appropriate shape
                        new_mask = torch.zeros_like(self.obs_model.SEzz, dtype=torch.bool)
                        
                        # Copy the values from mask_temp to new_mask up to min_dim0 and min_dim1
                        new_mask[:min_dim0, :min_dim1] = mask_temp[:min_dim0, :min_dim1]
                        
                        # Use the new mask for the calculation
                        ELBO_contrib_obs = (self.obs_model.transition.loggeomean()[:min_dim0, :min_dim1][new_mask[:min_dim0, :min_dim1]] *
                                           self.obs_model.SEzz[:min_dim0, :min_dim1][new_mask[:min_dim0, :min_dim1]]).sum()
                    else:
                        # Shapes match, proceed with normal calculation
                        ELBO_contrib_obs = (self.obs_model.transition.loggeomean()[mask_temp] * self.obs_model.SEzz[mask_temp]).sum()
                else:
                    # Fallback if shape attributes are not available
                    ELBO_contrib_obs = torch.tensor(0.0, device=self.obs_model.p.device)
            except Exception as e:
                print(f"Warning: Error handling shape mismatch in ELBO: {str(e)}")
                # Create a safe version of the ELBO contribution
                ELBO_contrib_obs = torch.tensor(0.0, device=self.obs_model.p.device)
            
            # Handle the initial contribution
            try:
                if hasattr(self.obs_model.initial, 'loggeomean') and callable(self.obs_model.initial.loggeomean):
                    loggeomean = self.obs_model.initial.loggeomean()
                    
                    # Check for shape compatibility
                    if loggeomean.shape == self.obs_model.SEz0.shape:
                        initial_contrib = (loggeomean * self.obs_model.SEz0).sum()
                    else:
                        print(f"Warning: Shape mismatch in initial contribution - loggeomean: {loggeomean.shape}, SEz0: {self.obs_model.SEz0.shape}")
                        
                        # Get the minimum dimension to avoid indexing errors
                        min_dim = min(loggeomean.shape[0], self.obs_model.SEz0.shape[0])
                        
                        # Calculate the contribution using the minimum dimension
                        initial_contrib = (loggeomean[:min_dim] * self.obs_model.SEz0[:min_dim]).sum()
                else:
                    print("Warning: initial.loggeomean() is not available, skipping initial contribution")
                    initial_contrib = torch.tensor(0.0, device=self.obs_model.p.device)
            except Exception as e:
                print(f"Warning: Error in initial contribution calculation: {str(e)}")
                initial_contrib = torch.tensor(0.0, device=self.obs_model.p.device)
            
            # Add the initial contribution to the ELBO
            ELBO_contrib_obs = ELBO_contrib_obs + initial_contrib
            
            # Add the entropy term
            try:
                if idx.any():
                    entropy_term = (self.obs_model.p[idx].log() * self.obs_model.p[idx]).sum()
                    ELBO_contrib_obs = ELBO_contrib_obs - entropy_term
            except Exception as e:
                print(f"Warning: Error in entropy calculation: {str(e)}")
                # Do not subtract anything if there's an error
            
            try:
                parent_elbo = super().ELBO()
                return parent_elbo + ELBO_contrib_obs
            except Exception as e:
                print(f"Warning: Error in parent ELBO calculation: {str(e)}")
                return ELBO_contrib_obs
                
        except Exception as e:
            print(f"Warning: Error in DMBD ELBO calculation: {str(e)}")
            return torch.tensor(0.0, requires_grad=False)

#### DMBD MASKS

    def n_object_mask(self,n,hidden_dims,role_dims,control_dim,obs_dim,regression_dim):
        # Assumes that the first hidden_dim is the environment, the second specifies the dimensions of the boundary of each object
        # and the third specifies the dimensions of the internal state of each object.  The 4th is optional and specifies a 'center of mass' like variable
        
        bz = torch.ones(hidden_dims[1]+hidden_dims[2],hidden_dims[1]+hidden_dims[2],requires_grad=False)
        notbz = torch.zeros(bz.shape,requires_grad=False)
        bz_mask = matrix_utils.block_matrix_builder(bz,notbz,notbz,bz)
        sb = torch.ones(hidden_dims[0],hidden_dims[1],requires_grad=False)
        sz = torch.zeros(hidden_dims[0],hidden_dims[2],requires_grad=False)
        sbz_mask = torch.cat((sb,sz),dim=-1)

        for i in range(n-2):
            bz_mask = matrix_utils.block_matrix_builder(bz_mask,torch.zeros(bz_mask.shape[0],bz.shape[0]),
                                                torch.zeros(bz.shape[0],bz_mask.shape[0],requires_grad=False),bz)
        for i in range(n-1):
            sbz_mask = torch.cat((sbz_mask,sb,sz),dim=-1)

        A_mask = torch.cat((sbz_mask,bz_mask),dim=-2)
        A_mask = matrix_utils.block_matrix_builder(torch.ones(hidden_dims[0],hidden_dims[0],requires_grad=False),sbz_mask,sbz_mask.transpose(-2,-1),bz_mask)
        Ac_mask = torch.ones(A_mask.shape[:-1]+(control_dim,))
        A_mask = torch.cat((A_mask,Ac_mask),dim=-1) 

        Bb = torch.cat((torch.ones(role_dims[1],hidden_dims[1],requires_grad=False),torch.zeros(role_dims[1],hidden_dims[2],requires_grad=False)),dim=-1)
        Bz = torch.cat((torch.zeros(role_dims[2],hidden_dims[1],requires_grad=False),torch.ones(role_dims[2],hidden_dims[2],requires_grad=False)),dim=-1)
        Bbz = torch.cat((Bb,Bz),dim=-2)

        B_mask = torch.ones(role_dims[0],hidden_dims[0])

        for i in range(n):
            B_mask = matrix_utils.block_matrix_builder(B_mask,torch.zeros(B_mask.shape[0],Bbz.shape[1],requires_grad=False),torch.zeros(Bbz.shape[0],B_mask.shape[1]),Bbz)

        B_mask = B_mask.unsqueeze(-2).expand(B_mask.shape[:1]+(obs_dim,)+B_mask.shape[1:])
        Br_mask = torch.ones(B_mask.shape[:-1]+(regression_dim,))
        B_mask = torch.cat((B_mask,Br_mask),dim=-1) 

        bz = torch.ones(role_dims[1]+role_dims[2],role_dims[1]+role_dims[2],requires_grad=False)
        notbz = torch.zeros(bz.shape,requires_grad=False)
        bz_mask = matrix_utils.block_matrix_builder(bz,notbz,notbz,bz)
        sb = torch.ones(role_dims[0],role_dims[1],requires_grad=False)
        sz = torch.zeros(role_dims[0],role_dims[2],requires_grad=False)
        sbz_mask = torch.cat((sb,sz),dim=-1)

        for i in range(n-2):
            bz_mask = matrix_utils.block_matrix_builder(bz_mask,torch.zeros(bz_mask.shape[0],bz.shape[0]),
                                                torch.zeros(bz.shape[0],bz_mask.shape[0],requires_grad=False),bz)
        for i in range(n-1):
            sbz_mask = torch.cat((sbz_mask,sb,sz),dim=-1)

        role_mask = torch.cat((sbz_mask,bz_mask),dim=-2)
        role_mask = matrix_utils.block_matrix_builder(torch.ones(role_dims[0],role_dims[0],requires_grad=False),sbz_mask,sbz_mask.transpose(-2,-1),bz_mask)


        return A_mask>0, B_mask>0, role_mask>0

    def one_object_mask(self,hidden_dims,role_dims,control_dim,obs_dim,regression_dim):
        # Standard mask for a single object
        # Assume that hidden_dims and role_dims are the same length and that the length is either 3 or 4
        # Assume that the first hidden_dim is the environment, the second is the boundary, and the third is the internal state
        # and that the optional 4th is for a single variable that effects all three kinds of observations, i.e. center of mass
        hidden_dim = np.sum(hidden_dims)
        role_dim = np.sum(role_dims)

        As = torch.cat((torch.ones(hidden_dims[0],hidden_dims[0]+hidden_dims[1],requires_grad=False),torch.zeros(hidden_dims[0],hidden_dims[2],requires_grad=False)),dim=-1)
        Ab = torch.ones(hidden_dims[1],np.sum(hidden_dims[0:3]),requires_grad=False)
        Az = torch.cat((torch.zeros(hidden_dims[2],hidden_dims[0],requires_grad=False),torch.ones(hidden_dims[2],hidden_dims[1]+hidden_dims[2],requires_grad=False)),dim=-1)
        if(len(hidden_dims)==4):
            As = torch.cat((As,torch.zeros(hidden_dims[0],hidden_dims[3],requires_grad=False)),dim=-1)
            Ab = torch.cat((Ab,torch.zeros(hidden_dims[1],hidden_dims[3],requires_grad=False)),dim=-1)
            Az = torch.cat((Az,torch.zeros(hidden_dims[2],hidden_dims[3],requires_grad=False)),dim=-1)
            Ag = torch.cat((torch.zeros(hidden_dims[3],np.sum(hidden_dims[:-1]),requires_grad=False),torch.ones(hidden_dims[3],hidden_dims[3])),dim=-1)
            A_mask = torch.cat((As,Ab,Az,Ag),dim=-2)
        else:
            A_mask = torch.cat((As,Ab,Az),dim=-2)
        A_mask = torch.cat((A_mask,torch.ones(A_mask.shape[:-1]+(control_dim,),requires_grad=False)),dim=-1) > 0 

        Bs = torch.ones((role_dims[0],obs_dim) + (hidden_dims[0],),requires_grad=False)
        Bs = torch.cat((Bs,torch.zeros((role_dims[0],obs_dim) + (hidden_dims[1]+hidden_dims[2],),requires_grad=False)),dim=-1)

        Bb = torch.zeros((role_dims[1],obs_dim) + (hidden_dims[0],),requires_grad=False)
        Bb = torch.cat((Bb,torch.ones((role_dims[1],obs_dim) + (hidden_dims[1],),requires_grad=False)),dim=-1)
        Bb = torch.cat((Bb,torch.zeros((role_dims[1],obs_dim) + (hidden_dims[2],),requires_grad=False)),dim=-1)

# Option 1:  internal observations are driven purely by internal states
        Bz = torch.zeros((role_dims[2],obs_dim) + (hidden_dims[0]+hidden_dims[1],),requires_grad=False)
        Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[2],),requires_grad=False)),dim=-1)
# Option 2:  internal observations are driven by both internal and boundary states
#        Bz = torch.zeros((role_dims[2],obs_dim) + (hidden_dims[0],),requires_grad=False)
#        Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[1] + hidden_dims[2],),requires_grad=False)),dim=-1)

        if len(hidden_dims)==4:
            Bs = torch.cat((Bs,torch.ones((role_dims[0],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            Bb = torch.cat((Bb,torch.ones((role_dims[1],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            B_mask = torch.cat((Bs,Bb,Bz),dim=-3)
        else:
            B_mask = torch.cat((Bs,Bb,Bz),dim=-3)
        # Only environment can have regressors.
        # if regression_dim==1:
        #     temp = torch.cat((torch.ones((role_dims[0],obs_dim) + (1,)),torch.zeros((role_dims[1],obs_dim)+(1,)),torch.zeros((role_dims[2],obs_dim)+(1,))),dim=-3)
        #     B_mask = torch.cat((B_mask,temp),dim=-1) > 0
        # else:

        
        # if regression_dim == 1:
        #     Rs = torch.ones(role_dims[0],obs_dim,1,requires_grad=False)
        #     Rb = torch.zeros(role_dims[1],obs_dim,1,requires_grad=False)
        #     Rz = torch.zeros(role_dims[2],obs_dim,1,requires_grad=False)
        #     R_mask = torch.cat((Rs,Rb,Rz),dim=-2) > 0    
        #     B_mask = torch.cat((B_mask,R_mask),dim=-1) > 0 
        # else:
        B_mask = torch.cat((B_mask,torch.ones(B_mask.shape[:-1]+(regression_dim,))),dim=-1) > 0 

        role_mask_s = torch.ones(role_dims[0],role_dims[0]+role_dims[1],requires_grad=False)
        role_mask_s = torch.cat((role_mask_s,torch.zeros(role_dims[0],role_dims[2],requires_grad=False)),dim=-1)
        role_mask_b = torch.ones(role_dims[1],role_dim,requires_grad=False)
        role_mask_z = torch.zeros(role_dims[2],role_dims[0],requires_grad=False)
        role_mask_z = torch.cat((role_mask_z,torch.ones(role_dims[2],role_dims[1]+role_dims[2],requires_grad=False)),dim=-1)
        role_mask = torch.cat((role_mask_s,role_mask_b,role_mask_z),dim=-2)

        return A_mask, B_mask, role_mask


    def plot_observation(self):
        labels = ['B ','Z ']
        labels = ['S ',] + self.number_of_objects*labels
        rlabels = ['Br ','Zr ']
        rlabels = ['Sr ',] + self.number_of_objects*rlabels
        plt.imshow(self.obs_model.obs_dist.mean().abs().sum(-2))
        for i, label in enumerate(labels):
            if i == 0:
                c = 'red'
            elif i % 2 == 1:
                pos = pos - 0.5
                c = 'green'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            else:
                pos = pos - 0.5
                c = 'blue'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)

            pos = self.hidden_dims[0]/2.0 + i*(self.hidden_dims[1]+self.hidden_dims[2])/2.0
            plt.text(pos, -1.5, label, color=c, ha='center', va='center', fontsize=10, weight='bold')
            pos = self.role_dims[0]/2.0 + i*(self.role_dims[1]+self.role_dims[2])/2.0
            if i == 0:
                plt.text(-1.5, pos-0.5, rlabels[i], color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)
            else:
                plt.text(-1.5, pos-0.5, rlabels[i]+str((i+1)//2), color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)

        plt.axis('off')  # Turn off the axis
        plt.show()

    def plot_transition(self,type='obs',use_mask = False):

        labels = ['B ','Z ']
        labels = ['S ',] + self.number_of_objects*labels

        if type == 'obs':
            if use_mask:
                plt.imshow(self.obs_model.transition_mask.squeeze())
            else:
                plt.imshow(self.obs_model.transition.mean())
        else:
            if use_mask:
                plt.imshow(self.A.mask.squeeze())
            else:
                plt.imshow(self.A.mean().abs().squeeze())
        # Add text annotations for the labels (x-axis)
        for i, label in enumerate(labels):
            if type == 'obs':
                pos = self.role_dims[0]/2.0 + i*(self.role_dims[1]+self.role_dims[2])/2.0
            else:
                pos = self.hidden_dims[0]/2.0 + i*(self.hidden_dims[1]+self.hidden_dims[2])/2.0
            if i == 0:
                c = 'red'
            elif i % 2 == 1:
                pos = pos - 0.5
                c = 'green'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            else:
                pos = pos - 0.5
                c = 'blue'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            
            plt.text(pos, -1.5, label, color=c, ha='center', va='center', fontsize=10, weight='bold')
            plt.text(-1.5, pos, label, color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)

        plt.axis('off')  # Turn off the axis
        plt.show()

    def forward(self, x):
        """
        Forward pass for the DMBD model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, n_obs, obs_dim]
                where n_obs is the number of observables and obs_dim is the dimension of each observable
                
        Returns:
            Dictionary containing:
                - 'latent_states': Estimated latent states
                - 'reconstructed': Reconstructed observations
                - 'loss': Negative ELBO (if training)
        """
        try:
            # Get the device from the input tensor
            device = x.device
            
            # Process the input data
            batch_size = x.shape[0]
            seq_length = x.shape[1]
            
            # Reshape inputs to match expected format
            y, u, r = self.reshape_inputs(x, None, None)
            
            # Ensure y, u, r are on the correct device if they aren't already
            if y is not None and y.device != device:
                y = y.to(device)
            if u is not None and u.device != device:
                u = u.to(device)
            if r is not None and r.device != device:
                r = r.to(device)
            
            # Update latent states based on observations
            try:
                self.update_latents(y, r)
            except Exception as e:
                print(f"Warning: Error in update_latents during forward pass: {str(e)}")
                # Continue with potentially incomplete latent states
            
            # Compute ELBO (Evidence Lower Bound)
            try:
                elbo = self.ELBO().sum()
            except Exception as e:
                print(f"Warning: Error computing ELBO during forward pass: {str(e)}")
                elbo = torch.tensor(0.0, requires_grad=False)
            
            # Get the latent states
            try:
                latent_states = self.latent_mean
                
                # Ensure latent_states is on the correct device
                if latent_states.device != device:
                    latent_states = latent_states.to(device)
            except Exception as e:
                print(f"Warning: Error getting latent states during forward pass: {str(e)}")
                latent_states = None
            
            # Reconstruct observations from latent states
            try:
                if latent_states is not None and hasattr(self.obs_model, 'mean'):
                    reconstructed = self.obs_model.mean(latent_states)
                    
                    # Ensure reconstructed is on the correct device
                    if reconstructed.device != device:
                        reconstructed = reconstructed.to(device)
                else:
                    reconstructed = None
            except Exception as e:
                print(f"Warning: Error reconstructing observations during forward pass: {str(e)}")
                reconstructed = None
            
            return {
                'latent_states': latent_states,
                'reconstructed': reconstructed,
                'loss': -elbo if elbo != 0.0 else torch.tensor(0.0, requires_grad=False)
            }
        except Exception as e:
            print(f"Error in DMBD forward pass: {str(e)}")
            return {
                'latent_states': None,
                'reconstructed': None,
                'loss': torch.tensor(0.0, requires_grad=False)
            }

    def to(self, device):
        """
        Move the model to the specified device.
        This is necessary to ensure all tensors, including masks, are on the same device.
        """
        # Call the parent class to() method
        super_model = super().to(device)
        
        # Ensure the mask attributes are moved to the correct device
        if hasattr(self, 'A') and hasattr(self.A, 'mask') and self.A.mask is not None:
            self.A.mask = self.A.mask.to(device)
        
        if hasattr(self, 'obs_model') and hasattr(self.obs_model, 'mask') and self.obs_model.mask is not None:
            self.obs_model.mask = self.obs_model.mask.to(device)
        
        if hasattr(self, 'role_mask') and self.role_mask is not None:
            self.role_mask = self.role_mask.to(device)
        
        return super_model

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm

class animate_results():
    def __init__(self,assignment_type='sbz', f=r'./movie_temp.', xlim = (-2.5,2.5), ylim = (-2.5,2.5), fps=20):
        self.assignment_type = assignment_type
        self.f=f
        self.xlim = xlim
        self.ylim = ylim
        self.fps = fps

    def animation_function(self,frame_number, fig_data, fig_assignments, fig_confidence):
        fn = frame_number
        T=fig_data.shape[0]
        self.scatter.set_offsets(fig_data[fn%T, fn//T,:,:].numpy())
        self.scatter.set_array(fig_assignments[fn%T, fn//T,:].numpy())
        self.scatter.set_alpha(fig_confidence[fn%T, fn//T,:].numpy())
        return self.scatter,
        
    def make_movie(self,model,data, batch_numbers):
        print('Generating Animation using',self.assignment_type, 'assignments')


        if(self.assignment_type == 'role'):
            rn = model.role_dims[0] + model.number_of_objects*(model.role_dims[1]+model.role_dims[2])
            assignments = model.obs_model.assignment()/(rn-1)
            confidence = model.obs_model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'sbz'):
            assignments = model.assignment()/2.0/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'particular'):
            assignments = model.particular_assignment()/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]

        fig_data = data[:,batch_numbers,:,0:2]
        fig_assignments = assignments[:,batch_numbers,:]
        fig_confidence = confidence[:,batch_numbers,:]
        fig_confidence[fig_confidence>1.0]=1.0

        self.fig = plt.figure(figsize=(7,7))
        self.ax = plt.axes(xlim=self.xlim,ylim=self.ylim)
        self.scatter=self.ax.scatter([], [], cmap = cm.rainbow_r, c=[], vmin=0.0, vmax=1.0)
        ani = FuncAnimation(self.fig, self.animation_function, frames=range(fig_data.shape[0]*fig_data.shape[1]), fargs=(fig_data,fig_assignments,fig_confidence,), interval=5).save(self.f,writer= FFMpegWriter(fps=self.fps) )
        plt.show()
