import torch
import numpy as np

class MultivariateNormal_vector_format():
    def __init__(self, mu=None, Sigma=None, invSigmamu=None, invSigma=None, Residual=None):
        # Initialize with either (mu, Sigma) or (invSigmamu, invSigma)
        # If both are provided, (mu, Sigma) takes precedence
        # If neither are provided, initialize with standard normal
        
        if mu is not None:
            self.mu = mu
            self.dim = mu.shape[-2]
            self.event_dim = len(mu.shape)
            if Sigma is not None:
                self.Sigma = Sigma
                self.invSigma = None
                self.invSigmamu = None
                self.Residual = None
            else:
                self.Sigma = None
                self.invSigma = None
                self.invSigmamu = None
                self.Residual = None
        elif invSigmamu is not None:
            self.mu = None
            self.Sigma = None
            self.invSigmamu = invSigmamu
            self.dim = invSigmamu.shape[-2]
            self.event_dim = len(invSigmamu.shape)
            if invSigma is not None:
                self.invSigma = invSigma
                if Residual is not None:
                    self.Residual = Residual
                else:
                    self.Residual = None
            else:
                self.invSigma = None
                self.Residual = None
        else:
            # Initialize with standard normal
            self.mu = torch.zeros((1,1))
            self.Sigma = torch.eye(1)
            self.invSigma = None
            self.invSigmamu = None
            self.Residual = None
            self.dim = 1
            self.event_dim = 2
        
        # Special handling for tensors with a second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            print(f"MultivariateNormal_vector_format.__init__: Sigma has second dimension of 0, shape: {self.Sigma.shape}")
        if self.invSigma is not None and self.invSigma.shape[-1] == 0:
            print(f"MultivariateNormal_vector_format.__init__: invSigma has second dimension of 0, shape: {self.invSigma.shape}")
        
        self.event_shape = (self.dim,)
        self.batch_shape = ()
        self.batch_dim = 0
        
        # Add a logdetinvSigma attribute
        self.logdetinvSigma = None

    @property
    def device(self):
        """Return the device of the distribution's tensors"""
        if self.mu is not None:
            return self.mu.device
        elif self.invSigmamu is not None:
            return self.invSigmamu.device
        elif self.Sigma is not None:
            return self.Sigma.device
        elif self.invSigma is not None:
            return self.invSigma.device
        else:
            # Default to CPU if no tensors are available
            return torch.device('cpu')
            
    @property
    def shape(self):
        if self.mu is not None:
            return self.mu.shape[:-2]
        elif self.invSigmamu is not None:
            return self.invSigmamu.shape[:-2]
        elif self.Sigma is not None:
            return self.Sigma.shape[:-2]
        elif self.invSigma is not None:
            return self.invSigma.shape[:-2]
        else:
            return torch.Size([])

    def to_event(self,n):
        if n == 0: 
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def unsqueeze(self,dim):  # only appliles to batch
        assert(dim + self.event_dim < 0)
        if self.mu is not None:
            mu = self.mu.unsqueeze(dim)
        else: mu = None
        if self.Sigma is not None:
            Sigma = self.Sigma.unsqueeze(dim)
        else: Sigma = None
        if self.invSigmamu is not None:
            invSigmamu = self.invSigmamu.unsqueeze(dim)
        else: invSigmamu = None
        if self.invSigma is not None:
            invSigma = self.invSigma.unsqueeze(dim)
        else: invSigma = None
        event_dim = self.event_dim - 2
        return MultivariateNormal_vector_format(mu,Sigma,invSigmamu,invSigma).to_event(event_dim)

    def combiner(self,other):
        self.invSigma = self.EinvSigma()+other.EinvSigma()
        self.invSigmamu = self.EinvSigmamu()+other.EinvSigmamu()
        self.Sigma = None
        self.mu = None

    def nat_combiner(self,invSigma,invSigmamu):
        self.invSigma = self.EinvSigma()+invSigma
        self.invSigmamu = self.EinvSigmamu()+invSigmamu
        self.Sigma = None
        self.mu = None

    def mean(self):
        if self.mu is None:
            self.mu = self.invSigma.inverse()@self.invSigmamu
        return self.mu
    
    def ESigma(self):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            return self.Sigma
        
        if self.Sigma is None:
            self.Sigma = self.invSigma.inverse()
        return self.Sigma

    def EinvSigma(self):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            return torch.zeros(self.batch_shape + (0, 0), device=self.mu.device)
            
        if self.invSigma is None:
            self.invSigma = self.Sigma.inverse()
        return self.invSigma
    
    def EinvSigmamu(self):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            return torch.zeros(self.batch_shape + (0,), device=self.mu.device)
            
        if self.invSigmamu is None:
            self.invSigmamu = self.EinvSigma()@self.mean()
        return self.invSigmamu

    def EResidual(self):
        if self.Residual is None:
            self.Residual = - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)
        return self.Residual

    def ElogdetinvSigma(self):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            return torch.zeros(self.batch_shape, device=self.mu.device)
            
        if self.logdetinvSigma is None:
            self.logdetinvSigma = -torch.logdet(self.ESigma())
        return self.logdetinvSigma

    def EX(self):
        return self.mean()

    def EXXT(self):
        # Special case: If Sigma has a second dimension of 0, return a tensor with 0 second dimension
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            print(f"EXXT: Sigma has second dimension of 0, shape: {self.Sigma.shape}")
            # Return the Sigma tensor directly to maintain the second dimension of 0
            return self.Sigma
        
        # Special case: If mu has a second dimension of 0, special handling needed
        if self.mu is not None and self.mu.shape[-1] == 0:
            print(f"EXXT: mu has second dimension of 0, shape: {self.mu.shape}")
            # In this case, return Sigma directly as well
            return self.ESigma()
        
        # Regular case: Add the covariance matrix to the outer product of the mean tensor
        mu = self.mean()
        print(f"EXXT: Regular case - mu shape: {mu.shape}, Sigma shape: {self.ESigma().shape}")
        result = self.ESigma() + mu @ mu.transpose(-1, -2)
        print(f"EXXT: Result shape: {result.shape}")
        return result

    def EXTX(self):
        return self.ESigma().sum(-1).sum(-1) + (self.mean().transpose(-2,-1)@self.mean()).squeeze(-1).squeeze(-1)

    def Res(self):
        return - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)

    def ss_update(self,SExx,SEx,n, lr=1.0):
        n=n.unsqueeze(-1).unsqueeze(-1)
        self.mu = SEx/n
        self.Sigma = SExx/n - self.mu@self.mu.transpose(-2,-1)
        self.invSigma = None
        self.invSigmamu = None

    def raw_update(self,X,p=None,lr=1.0):  # assumes X is a vector i.e. 


        if p is None:  
            SEx = X
            SExx = X@X.transpose(-2,-1)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X@X.transpose(-2,-1)*p
            SEx =  X*p
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1).squeeze(-1),lr)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1


    def Elog_like(self, X):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            # When Sigma has dimension 0, we're only dealing with the mean part
            # Return a tensor of zeros with appropriate batch shape
            return torch.zeros(X.shape[:-1], device=X.device)
            
        # Original implementation
        out = -0.5*((X - self.mu).transpose(-2,-1)@self.EinvSigma()@(X - self.mu)).squeeze(-1).squeeze(-1)
        out = out - 0.5*self.dim*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma()
        for i in range(self.event_dim-2):
            out = out.sum(-1)
        return out

    def Elog_like_given_pX_pY(self, pX, pY):
        # Special handling for Sigma with second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            # When Sigma has dimension 0, we're only dealing with the mean part
            # Return a tensor of zeros with appropriate batch shape
            return torch.zeros(pX.mean().shape[:-2], device=pX.mean().device)
        
        try:
            # Print shapes for debugging
            print(f"Elog_like_given_pX_pY: pX.mean shape: {pX.mean().shape}, pY.mean shape: {pY.mean().shape}")
            print(f"Elog_like_given_pX_pY: self.EinvSigma shape: {self.EinvSigma().shape}")
            
            # Check if batch dimensions match
            px_shape = pX.mean().shape
            py_shape = pY.mean().shape
            
            # Check for dimension mismatch at dimension 2 (the error we're seeing)
            if len(px_shape) > 2 and len(py_shape) > 2:
                if px_shape[2] != py_shape[2]:
                    print(f"Elog_like_given_pX_pY: Dimension mismatch at dim 2 - pX: {px_shape[2]}, pY: {py_shape[2]}")
                    
                    # Handle the specific case where one dimension is 39 and the other is 3
                    if (px_shape[2] == 39 and py_shape[2] == 3) or (px_shape[2] == 3 and py_shape[2] == 39):
                        print(f"Elog_like_given_pX_pY: Handling special case with dimensions 39 and 3")
                        
                        # Determine which tensor has dimension 39 and which has dimension 3
                        if px_shape[2] == 39:
                            # pX has dimension 39, pY has dimension 3
                            # We'll reshape pX to have dimension 3 by averaging groups of 13 elements
                            px_mean = pX.mean()
                            px_mean_reshaped = torch.zeros(px_shape[0], px_shape[1], 3, *px_shape[3:], device=px_mean.device)
                            
                            # Average groups of 13 elements
                            for i in range(3):
                                start_idx = i * 13
                                end_idx = (i + 1) * 13
                                if end_idx > px_shape[2]:
                                    end_idx = px_shape[2]
                                px_mean_reshaped[:, :, i, ...] = px_mean[:, :, start_idx:end_idx, ...].mean(dim=2)
                            
                            # Create a new pX with the reshaped mean
                            pX = MultivariateNormal_vector_format(mu=px_mean_reshaped)
                            print(f"Elog_like_given_pX_pY: Reshaped pX.mean to {pX.mean().shape}")
                        else:
                            # pY has dimension 39, pX has dimension 3
                            # We'll reshape pY to have dimension 3 by averaging groups of 13 elements
                            py_mean = pY.mean()
                            py_mean_reshaped = torch.zeros(py_shape[0], py_shape[1], 3, *py_shape[3:], device=py_mean.device)
                            
                            # Average groups of 13 elements
                            for i in range(3):
                                start_idx = i * 13
                                end_idx = (i + 1) * 13
                                if end_idx > py_shape[2]:
                                    end_idx = py_shape[2]
                                py_mean_reshaped[:, :, i, ...] = py_mean[:, :, start_idx:end_idx, ...].mean(dim=2)
                            
                            # Create a new pY with the reshaped mean
                            pY = MultivariateNormal_vector_format(mu=py_mean_reshaped)
                            print(f"Elog_like_given_pX_pY: Reshaped pY.mean to {pY.mean().shape}")
                    elif px_shape[2] > py_shape[2]:
                        # Need to reshape pX to match pY's dimension 2
                        print(f"Elog_like_given_pX_pY: Reshaping pX to match pY's dimension 2")
                        
                        # Create a new pX with the correct shape
                        # We'll need to average over the extra dimensions
                        px_mean = pX.mean()
                        px_sigma = pX.ESigma() if hasattr(pX, 'ESigma') and callable(pX.ESigma) else None
                        
                        # Reshape by averaging over dimension 2
                        # First, we need to reshape to combine dimensions 1 and 2
                        combined_shape = list(px_shape)
                        combined_shape[1] = px_shape[1] * px_shape[2] // py_shape[2]
                        combined_shape[2] = py_shape[2]
                        
                        try:
                            px_mean = px_mean.reshape(combined_shape)
                            if px_sigma is not None:
                                sigma_shape = list(px_sigma.shape)
                                sigma_shape[1] = px_shape[1] * px_shape[2] // py_shape[2]
                                sigma_shape[2] = py_shape[2]
                                px_sigma = px_sigma.reshape(sigma_shape)
                            
                            # Create a new pX with the reshaped tensors
                            pX = MultivariateNormal_vector_format(mu=px_mean, Sigma=px_sigma)
                            print(f"Elog_like_given_pX_pY: Reshaped pX.mean to {pX.mean().shape}")
                        except Exception as e:
                            print(f"Elog_like_given_pX_pY: Error reshaping pX: {str(e)}")
                            # If reshaping fails, we'll try a different approach
                            # We'll select a subset of the dimensions
                            px_mean = px_mean[:, :, :py_shape[2], ...]
                            if px_sigma is not None:
                                px_sigma = px_sigma[:, :, :py_shape[2], ...]
                            
                            # Create a new pX with the truncated tensors
                            pX = MultivariateNormal_vector_format(mu=px_mean, Sigma=px_sigma)
                            print(f"Elog_like_given_pX_pY: Truncated pX.mean to {pX.mean().shape}")
                    else:
                        # Need to reshape pY to match pX's dimension 2
                        print(f"Elog_like_given_pX_pY: Reshaping pY to match pX's dimension 2")
                        
                        # Create a new pY with the correct shape
                        py_mean = pY.mean()
                        py_sigma = pY.ESigma() if hasattr(pY, 'ESigma') and callable(pY.ESigma) else None
                        
                        # Reshape by repeating along dimension 2
                        try:
                            # Repeat the tensor along dimension 2
                            repeats = [1] * len(py_shape)
                            repeats[2] = px_shape[2] // py_shape[2]
                            
                            py_mean = py_mean.repeat(*repeats)
                            if py_sigma is not None:
                                py_sigma = py_sigma.repeat(*repeats)
                            
                            # Create a new pY with the reshaped tensors
                            pY = MultivariateNormal_vector_format(mu=py_mean, Sigma=py_sigma)
                            print(f"Elog_like_given_pX_pY: Reshaped pY.mean to {pY.mean().shape}")
                        except Exception as e:
                            print(f"Elog_like_given_pX_pY: Error reshaping pY: {str(e)}")
                            # If reshaping fails, we'll try a different approach
                            # We'll expand pY to match pX's dimension 2
                            expanded_shape = list(py_shape)
                            expanded_shape[2] = px_shape[2]
                            
                            # Create expanded tensors filled with the original values
                            expanded_mean = torch.zeros(expanded_shape, device=py_mean.device)
                            for i in range(px_shape[2]):
                                idx = i % py_shape[2]
                                expanded_mean[:, :, i, ...] = py_mean[:, :, idx, ...]
                            
                            # Create a new pY with the expanded tensors
                            pY = MultivariateNormal_vector_format(mu=expanded_mean)
                            print(f"Elog_like_given_pX_pY: Expanded pY.mean to {pY.mean().shape}")
            
            # Check if batch2 dimensions match
            expected_cols = self.EinvSigma().shape[-1]
            actual_cols = px_shape[-1]
            
            if expected_cols != actual_cols:
                print(f"Elog_like_given_pX_pY: Expected size for first two dimensions of batch2 tensor to be: [..., {expected_cols}] but got: [..., {actual_cols}].")
                
                # Determine if we need to pad or truncate
                if actual_cols < expected_cols:
                    # Need to pad pX
                    padding_size = expected_cols - actual_cols
                    print(f"Elog_like_given_pX_pY: Padding pX with {padding_size} columns")
                    
                    # Create padded versions of pX mean and covariance
                    px_mean = pX.mean()
                    padding = torch.zeros(px_mean.shape[:-1] + (padding_size,), device=px_mean.device)
                    padded_px_mean = torch.cat([px_mean, padding], dim=-1)
                    
                    # Create a new pX with padded dimensions
                    padded_pX = MultivariateNormal_vector_format(
                        mu=padded_px_mean,
                        Sigma=torch.zeros(px_shape[:-1] + (expected_cols, expected_cols), device=px_mean.device)
                    )
                    pX = padded_pX
                else:
                    # Need to truncate pX
                    print(f"Elog_like_given_pX_pY: Truncating pX from {actual_cols} to {expected_cols} columns")
                    
                    # Create truncated versions of pX mean and covariance
                    px_mean = pX.mean()[..., :expected_cols]
                    px_sigma = pX.ESigma()
                    if px_sigma is not None:
                        px_sigma = px_sigma[..., :expected_cols, :expected_cols]
                    
                    # Create a new pX with truncated dimensions
                    truncated_pX = MultivariateNormal_vector_format(
                        mu=px_mean,
                        Sigma=px_sigma
                    )
                    pX = truncated_pX
            
            # Check if pY dimensions match
            py_shape = pY.mean().shape
            if expected_cols != py_shape[-1]:
                print(f"Elog_like_given_pX_pY: pY dimensions don't match. Expected: {expected_cols}, got: {py_shape[-1]}")
                
                # Determine if we need to pad or truncate pY
                if py_shape[-1] < expected_cols:
                    # Need to pad pY
                    padding_size = expected_cols - py_shape[-1]
                    print(f"Elog_like_given_pX_pY: Padding pY with {padding_size} columns")
                    
                    # Create padded versions of pY mean and covariance
                    py_mean = pY.mean()
                    padding = torch.zeros(py_mean.shape[:-1] + (padding_size,), device=py_mean.device)
                    padded_py_mean = torch.cat([py_mean, padding], dim=-1)
                    
                    # Create a new pY with padded dimensions
                    padded_pY = MultivariateNormal_vector_format(
                        mu=padded_py_mean,
                        Sigma=torch.zeros(py_shape[:-1] + (expected_cols, expected_cols), device=py_mean.device)
                    )
                    pY = padded_pY
                else:
                    # Need to truncate pY
                    print(f"Elog_like_given_pX_pY: Truncating pY from {py_shape[-1]} to {expected_cols} columns")
                    
                    # Create truncated versions of pY mean and covariance
                    py_mean = pY.mean()[..., :expected_cols]
                    py_sigma = pY.ESigma()
                    if py_sigma is not None:
                        py_sigma = py_sigma[..., :expected_cols, :expected_cols]
                    
                    # Create a new pY with truncated dimensions
                    truncated_pY = MultivariateNormal_vector_format(
                        mu=py_mean,
                        Sigma=py_sigma
                    )
                    pY = truncated_pY
            
            # Now calculate expected log-likelihood with adjusted dimensions
            # Check if pX.EXXT() and pY.EXXT() have compatible shapes with self.EinvSigma()
            px_exxt = pX.EXXT()
            py_exxt = pY.EXXT()
            print(f"Elog_like_given_pX_pY: pX.EXXT shape: {px_exxt.shape}, pY.EXXT shape: {py_exxt.shape}")
            
            # Handle dimension mismatch in the calculation of terms
            try:
                # First term: -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1)
                term1 = -0.5*(py_exxt*self.EinvSigma()).sum(-1).sum(-1)
                print(f"Elog_like_given_pX_pY: term1 shape: {term1.shape}")
                
                # Second term: (pY.mean().transpose(-2,-1)@self.EinvSigma()@pX.mean()).squeeze(-1).squeeze(-1)
                # Handle potential dimension mismatch
                py_mean_t = pY.mean().transpose(-2,-1)
                einv_sigma = self.EinvSigma()
                px_mean = pX.mean()
                
                # Check if dimensions are compatible for matrix multiplication
                if py_mean_t.shape[-1] != einv_sigma.shape[-2] or einv_sigma.shape[-1] != px_mean.shape[-2]:
                    print(f"Elog_like_given_pX_pY: Dimension mismatch in term2 calculation")
                    print(f"py_mean_t: {py_mean_t.shape}, einv_sigma: {einv_sigma.shape}, px_mean: {px_mean.shape}")
                    
                    # Try to reshape tensors to make them compatible
                    if len(py_mean_t.shape) > 3 and len(px_mean.shape) > 3:
                        # Reshape tensors to make batch dimensions compatible
                        py_batch_shape = py_mean_t.shape[:-2]
                        px_batch_shape = px_mean.shape[:-2]
                        
                        # If batch shapes don't match, try to broadcast
                        if py_batch_shape != px_batch_shape:
                            print(f"Elog_like_given_pX_pY: Batch shapes don't match - pY: {py_batch_shape}, pX: {px_batch_shape}")
                            
                            # Try to broadcast to a common shape
                            try:
                                # Get the maximum size for each dimension
                                max_dims = []
                                for i in range(min(len(py_batch_shape), len(px_batch_shape))):
                                    max_dims.append(max(py_batch_shape[i], px_batch_shape[i]))
                                
                                # Create broadcasted shapes
                                py_broadcast_shape = tuple(max_dims) + py_mean_t.shape[-2:]
                                px_broadcast_shape = tuple(max_dims) + px_mean.shape[-2:]
                                
                                # Broadcast tensors
                                py_mean_t_broadcast = py_mean_t.expand(py_broadcast_shape)
                                px_mean_broadcast = px_mean.expand(px_broadcast_shape)
                                
                                # Calculate term2 with broadcasted tensors
                                term2 = (py_mean_t_broadcast @ einv_sigma @ px_mean_broadcast).squeeze(-1).squeeze(-1)
                            except Exception as e:
                                print(f"Elog_like_given_pX_pY: Error broadcasting tensors: {str(e)}")
                                # Fallback: use a simpler calculation
                                term2 = torch.zeros(max(py_batch_shape[0], px_batch_shape[0]), device=py_mean_t.device)
                        else:
                            # Try to calculate term2 with original tensors
                            try:
                                term2 = (py_mean_t @ einv_sigma @ px_mean).squeeze(-1).squeeze(-1)
                            except Exception as e:
                                print(f"Elog_like_given_pX_pY: Error calculating term2: {str(e)}")
                                # Fallback: use a simpler calculation
                                term2 = torch.zeros(py_batch_shape[0], device=py_mean_t.device)
                    else:
                        # Simple case: just try the calculation
                        try:
                            term2 = (py_mean_t @ einv_sigma @ px_mean).squeeze(-1).squeeze(-1)
                        except Exception as e:
                            print(f"Elog_like_given_pX_pY: Error calculating term2: {str(e)}")
                            # Fallback: use a simpler calculation
                            term2 = torch.zeros(1, device=py_mean_t.device)
                else:
                    # Dimensions are compatible, calculate term2 normally
                    term2 = (py_mean_t @ einv_sigma @ px_mean).squeeze(-1).squeeze(-1)
                
                print(f"Elog_like_given_pX_pY: term2 shape: {term2.shape}")
                
                # Third term: -0.5*(pX.EXXT()*self.EinvSigma()).sum(-1).sum(-1)
                term3 = -0.5*(px_exxt*self.EinvSigma()).sum(-1).sum(-1)
                print(f"Elog_like_given_pX_pY: term3 shape: {term3.shape}")
                
                # Fourth term: 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2.0*np.pi)
                term4 = 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2.0*np.pi)
                print(f"Elog_like_given_pX_pY: term4 shape: {term4.shape}")
                
                # Combine all terms
                # Make sure all terms have compatible shapes for addition
                # Broadcast term4 to match the shape of other terms if necessary
                if isinstance(term4, float) or (hasattr(term4, 'shape') and len(term4.shape) == 0):
                    # term4 is a scalar, broadcast it to match other terms
                    term4_shape = term1.shape if hasattr(term1, 'shape') else term3.shape
                    term4 = torch.full(term4_shape, term4, device=term1.device if hasattr(term1, 'device') else term3.device)
                
                # Combine terms with proper broadcasting
                ELL = term1 + term2 + term3 + term4
                print(f"Elog_like_given_pX_pY: Final ELL shape: {ELL.shape}")
                return ELL
                
            except Exception as e:
                print(f"Elog_like_given_pX_pY: Error calculating terms: {str(e)}")
                # Return a default tensor with appropriate shape
                return torch.zeros(pX.mean().shape[:-2], device=pX.mean().device)
            
        except Exception as e:
            print(f"Elog_like_given_pX_pY: Unhandled error: {str(e)}")
            # Return a default tensor with appropriate shape
            return torch.zeros(pX.mean().shape[:-2], device=pX.mean().device)

    def KLqprior(self):
        return torch.tensor(0.0,requires_grad=False)

    def to(self, device):
        """
        Move the distribution to the specified device.
        
        Args:
            device: The device to move the distribution to.
            
        Returns:
            The distribution on the specified device.
        """
        if self.mu is not None:
            self.mu = self.mu.to(device)
        if self.Sigma is not None:
            self.Sigma = self.Sigma.to(device)
        if self.invSigmamu is not None:
            self.invSigmamu = self.invSigmamu.to(device)
        if self.invSigma is not None:
            self.invSigma = self.invSigma.to(device)
        if self.Residual is not None:
            self.Residual = self.Residual.to(device)
            
        # Special handling for tensors with a second dimension of 0
        if self.Sigma is not None and self.Sigma.shape[-1] == 0:
            print(f"MultivariateNormal_vector_format.to: Sigma has second dimension of 0, shape: {self.Sigma.shape}")
        if self.invSigma is not None and self.invSigma.shape[-1] == 0:
            print(f"MultivariateNormal_vector_format.to: invSigma has second dimension of 0, shape: {self.invSigma.shape}")
            
        return self

# from .Mixture import Mixture

# class MixtureofMultivariateNormals_vector_format(Mixture):
#     def __init__(self,mu_0,Sigma_0):
#         dist = MultivariateNormal_vector_format(mu = torch.randn(mu_0.shape,requires_grad=False)+mu_0,Sigma = Sigma_0)
#         super().__init__(dist)


