# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 
import torch
import numpy as np
from .DiagonalWishart import DiagonalWishart, DiagonalWishart_UnitTrace
from .utils import matrix_utils
from .MultivariateNormal_vector_format import MultivariateNormal_vector_format

class MatrixNormalGamma():
    # Conugate prior for linear regression coefficients
    # mu is assumed to be n x p
    # V is assumed to be p x p and plays the role of lambda for the normal wishart prior
    # invU is n x n diagonal matrix with diagonal elements assumed to be gamma distributed
    # i.e.  Y = A @ X + U^{-1/2} @ eps, where A is the random variable represented by the MNW prior
    # When used for linear regression, either zscore X and Y or pad X with a column of ones
    #   mask is a boolean tensor of shape mu_0 that indicates which entries of mu can be non-zero
    #         mask must be the same for all elements of a given batch, i.e. it is n x p 
    #   X_mask is a boolean tensor of shape (mu_0.shape[:-2] + (1,) + mu_shape[-1:]) 
    #           that indicates which entries of X contribute to the prediction 

    def __init__(self,mu_0,U_0=None,V_0=None,uniform_precision=False,mask=None,X_mask=None,pad_X=False):
        self.n = mu_0.shape[-2]
        self.p = mu_0.shape[-1]
        self.pad_X = pad_X
        if pad_X:
            self.p = self.p+1
            mu_0 = torch.cat((mu_0,torch.zeros(mu_0.shape[:-1]+(1,),requires_grad=False)),dim=-1)
        self.event_dim = 2  # this is the dimension of the observation Y in Y = AX
        self.batch_dim = mu_0.ndim  - 2
        self.event_shape = mu_0.shape[-2:]
        self.batch_shape = mu_0.shape[:-2]
        if self.batch_dim == 0:
            self.batch_shape = torch.Size(())
        if U_0 is None:
            U_0 = torch.zeros(self.batch_shape + (self.n,),requires_grad=False) + torch.ones(self.n,requires_grad=False)
        elif U_0.shape[-2] == U_0.shape[-1]:
            U_0 = U_0.diagonal(dim1=-2,dim2=-1)
        if V_0 is None:
            V_0 = torch.zeros(self.batch_shape + (self.p,self.p),requires_grad=False) + torch.eye(self.p,requires_grad=False)
        elif pad_X:
            temp = V_0.sum()/np.prod(V_0.shape[:-1])
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-1]+(1,),requires_grad=False)),dim=-1)
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-2]+(1,)+V_0.shape[-1:],requires_grad=False)),dim=-2)
            V_0[...,-1,-1] = temp

        self.uniform_precision = uniform_precision
        self.mask = mask
        self.X_mask = X_mask
        self.mu_0 = mu_0
        self.mu = 0.1*torch.randn(mu_0.shape,requires_grad=False)/np.sqrt(self.n*self.p)+mu_0
        self.invV_0 = V_0.inverse()
        self.V = V_0
        self.invV = self.invV_0
        self.invU = DiagonalWishart(2*torch.ones(U_0.shape,requires_grad=False),2*U_0)
        self.logdetinvV = self.invV.logdet()    
        self.logdetinvV_0=self.invV_0.logdet()    

        if X_mask is not None:
            if pad_X:
                self.X_mask = torch.cat((X_mask,torch.ones(X_mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            # Ensure X_mask and mu_0 have compatible shapes before applying mask
            if self.X_mask.shape[-1] != self.mu_0.shape[-1]:
                # Adjust mu_0 to match X_mask's last dimension
                if self.X_mask.shape[-1] < self.mu_0.shape[-1]:
                    self.mu_0 = self.mu_0[..., :self.X_mask.shape[-1]]
                    self.mu = self.mu[..., :self.X_mask.shape[-1]]
                else:
                    # This case shouldn't happen in practice, but handle it anyway
                    pad_size = self.X_mask.shape[-1] - self.mu_0.shape[-1]
                    self.mu_0 = torch.cat((self.mu_0, torch.zeros(self.mu_0.shape[:-1] + (pad_size,), requires_grad=False)), dim=-1)
                    self.mu = torch.cat((self.mu, torch.zeros(self.mu.shape[:-1] + (pad_size,), requires_grad=False)), dim=-1)
            
            self.mu_0 = self.mu_0*self.X_mask
            self.mu = self.mu*self.X_mask
            self.V = self.V*self.X_mask*self.X_mask.transpose(-2,-1)
            self.invV = self.invV*self.X_mask*self.X_mask.transpose(-2,-1)

        if mask is not None:
            if pad_X:
                self.mask = torch.cat((self.mask,torch.ones(self.mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            
            # Ensure mask and mu_0 have compatible shapes before applying mask
            if self.mask.shape[-1] != self.mu_0.shape[-1]:
                # Adjust mu_0 to match mask's last dimension
                if self.mask.shape[-1] < self.mu_0.shape[-1]:
                    self.mu_0 = self.mu_0[..., :self.mask.shape[-1]]
                    self.mu = self.mu[..., :self.mask.shape[-1]]
                else:
                    # This case shouldn't happen in practice, but handle it anyway
                    pad_size = self.mask.shape[-1] - self.mu_0.shape[-1]
                    self.mu_0 = torch.cat((self.mu_0, torch.zeros(self.mu_0.shape[:-1] + (pad_size,), requires_grad=False)), dim=-1)
                    self.mu = torch.cat((self.mu, torch.zeros(self.mu.shape[:-1] + (pad_size,), requires_grad=False)), dim=-1)
            
            self.mu_0 = self.mu_0*self.mask
            self.mu = self.mu*self.mask

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def to(self, device):
        """
        Move the distribution to the specified device.
        
        Args:
            device: The device to move the distribution to.
            
        Returns:
            The distribution on the specified device.
        """
        # Move all tensor attributes to the target device
        if hasattr(self, 'mu') and self.mu is not None:
            self.mu = self.mu.to(device)
            
        if hasattr(self, 'mu_0') and self.mu_0 is not None:
            self.mu_0 = self.mu_0.to(device)
            
        if hasattr(self, 'V') and self.V is not None:
            self.V = self.V.to(device)
            
        if hasattr(self, 'invV') and self.invV is not None:
            self.invV = self.invV.to(device)
            
        if hasattr(self, 'invV_0') and self.invV_0 is not None:
            self.invV_0 = self.invV_0.to(device)
            
        if hasattr(self, 'invU') and self.invU is not None:
            self.invU = self.invU.to(device)
            
        if hasattr(self, 'mask') and self.mask is not None:
            self.mask = self.mask.to(device)
            
        if hasattr(self, 'X_mask') and self.X_mask is not None:
            self.X_mask = self.X_mask.to(device)
            
        return self

    def ss_update(self,SExx,SEyx,SEyy,n,lr=1.0):

        if self.X_mask is not None:
            SExx = SExx*self.X_mask*self.X_mask.transpose(-2,-1)
            SEyx = SEyx*self.X_mask
            invV = self.invV_0 + SExx
            muinvV = self.mu_0@self.invV_0 + SEyx
            mu = muinvV@invV.inverse()
            mu = mu*self.X_mask
        else:
            invV = self.invV_0 + SExx
            muinvV = self.mu_0@self.invV_0 + SEyx
            mu = torch.linalg.solve(invV,muinvV.transpose(-2,-1)).transpose(-2,-1)
            # mu = muinvV@invV.inverse()

        if self.mask is not None:  # Assumes mask is same for whole batch
            V = invV.inverse()
            U = self.invU.EinvSigma().inverse()
            Astar = V.unsqueeze(-3).unsqueeze(-2)*U.unsqueeze(-2).unsqueeze(-1)
            A  = Astar[...,~self.mask,:,:][...,:,~self.mask]            
            b = mu[...,~self.mask]
            gamma = torch.zeros_like(mu)
            gamma[...,~self.mask] = torch.linalg.solve(A,b)
            mu = mu - U@gamma@V
            mu = mu*self.mask

        SEyy = SEyy - mu@invV@mu.transpose(-2,-1)
        SEyy = SEyy + self.mu_0@self.invV_0@self.mu_0.transpose(-2,-1)
        self.invU.ss_update(SEyy.diagonal(dim1=-2,dim2=-1), n.unsqueeze(-1), lr)
        self.invV = (invV-self.invV)*lr + self.invV
        self.invV = 0.5*(self.invV + self.invV.transpose(-2,-1))
        self.mu = (mu-self.mu)*lr + self.mu
        if(self.mask is not None):
            self.mu = self.mu*self.mask

#        self.invV_d, self.invV_v = torch.linalg.eigh(self.invV) 
#        self.V = self.invV_v@(1.0/self.invV_d.unsqueeze(-1)*self.invV_v.transpose(-2,-1))
#        self.logdetinvV = self.invV_d.log().sum(-1)
        self.V = self.invV.inverse()
        self.logdetinvV = self.invV.logdet()

        if self.X_mask is not None:
#            self.V = self.V * self.X_mask * self.X_mask.transpose(-2,-1)
#            self.invV = self.invV * self.X_mask * self.X_mask.transpose(-2,-1)
            self.mu = self.mu*self.X_mask
#            self.logdetinvV = self.logdetinvV - self.logdetinvV_0*(~self.X_mask).sum(-1).sum(-1)

        if self.uniform_precision==True:
            self.invU.gamma.alpha = self.invU.gamma.alpha.sum(-1,keepdim=True)  # THIS IS A HACK

    def update(self,pX,pY,p=None,lr=1.0):
        if p is None:
            sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
            SExx = pX.EXXT().sum(0)
            SEyy = pY.EXXT().sum(0)
            SEyx = (pY.EX()@pX.EX().transpose(-2,-1)).sum(0)
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)

            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)

            if self.pad_X:
                SEx = pX.EX().sum(0)
                SEy = pY.EX().sum(0)
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)
                    
                SExx = torch.cat((SExx,SEx),dim=-1)
                SEx = torch.cat((SEx,n.expand(SEx.shape[:-2]+(1,1))),dim=-2)
                SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
                SEyx = torch.cat((SEyx,SEy.expand(SEyx.shape[:-1]+(1,))),dim=-1)

            n = n.expand(self.batch_shape + self.event_shape[:-2])
            self.ss_update(SExx,SEyx,SEyy,n,lr)
            
        else:
            p=p.view(p.shape + self.event_dim*(1,))
            SExx = (pX.EXXT()*p).sum(0)
            SEyy = (pY.EXXT()*p).sum(0)
            SEyx = ((pY.EX()@pX.EX().transpose(-2,-1))*p).sum(0)
            if self.pad_X:
                SEx = (pX.EX()*p).sum(0)
                SEy = (pY.EX()*p).sum(0)
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)
            p=p.sum(0)
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
                p=p.sum(0)

            if self.pad_X:
                SExx = torch.cat((SExx,SEx),dim=-1)
                SEx = torch.cat((SEx,p),dim=-2)
                SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
                SEyx = torch.cat((SEyx,SEy),dim=-1)
            self.ss_update(SExx,SEyx,SEyy,p.view(p.shape[:-2]),lr)

    def raw_update(self,X,Y,p=None,lr=1.0):
        if self.pad_X:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),dim=-2)
        if p is None: 
            sample_shape = X.shape[:-self.event_dim-self.batch_dim+1]
            SExx = X@X.transpose(-2,-1)
            SEyy = Y@Y.transpose(-2,-1)
            SEyx = Y@X.transpose(-2,-1)
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])

            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
            self.ss_update(SExx,SEyx,SEyy,n,lr)
            
        else:
            p = p.view(p.shape+self.event_dim*(1,))
            SExx = X@X.transpose(-2,-1)*p
            SEyy = Y@Y.transpose(-2,-1)*p
            SEyx = Y@X.transpose(-2,-1)*p
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
                p=p.sum(0)
            self.ss_update(SExx,SEyx,SEyy,p.view(p.shape[:-2]),lr)

    def KLqprior(self):

        KL = self.n/2.0*self.logdetinvV - self.n/2.0*self.logdetinvV_0 - self.n*self.p/2.0
        if self.X_mask is not None:
            KL = KL + self.n/2.0*self.logdetinvV_0*(self.X_mask).sum(-1).sum(-1)
        KL = KL + 0.5*self.n*(self.invV_0*self.V).sum(-1).sum(-1)
        temp = (self.mu-self.mu_0).transpose(-2,-1)@(self.invU.gamma.mean().unsqueeze(-1)*(self.mu-self.mu_0))
        KL = KL + 0.5*(self.invV_0*temp).sum(-1).sum(-1)
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        if self.uniform_precision == True:
            KL = KL + self.invU.KLqprior()/self.n
        else:
            KL = KL + self.invU.KLqprior()
        return KL

    def forward(self,pX):
        if self.pad_X:
            PJ_y_y = self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1] + pX.EinvSigma()
            PmuJ_y = self.EinvUX()[...,:,-1:]
            PmuJ_x = pX.EinvSigmamu()-self.EXTinvUX()[...,:-1,-1:]
#            PJ11 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX() + pX.EinvSigma()
            PmuJ_y = torch.zeros(PJ_y_y.shape[:-1]+(1,))
            PmuJ_x = pX.EinvSigmamu()
#            PJ11 = torch.tensor(0.0)

        invSigma_y_y, negBinvD = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)[0:2]
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        return MultivariateNormal_vector_format(invSigma = invSigma_y_y, invSigmamu = invSigmamu_y)

    def Elog_like(self,X,Y):  # expects X to be batch_shape* + event_shape[:-1] by p 
                              #     and Y to be batch_shape* + event_shape[:-1] by n
                              # for mixtures batch_shape* = batch_shape[:-1]+(1,)
        ELL = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1)
        if self.pad_X:
            ELL = ELL + (Y.transpose(-2,-1)@(self.EinvUX()[...,:,:-1]@X + self.EinvUX()[...,:,-1:])).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(X.transpose(-2,-1)@self.EXTinvUX()[...,:-1,:-1]@X + 2*self.EXTinvUX()[...,-1:,:-1]@X +  self.EXTinvUX()[...,-1:,-1:]).squeeze(-1).squeeze(-1)
        else:
            ELL = ELL + (Y.transpose(-2,-1)@self.EinvUX()@X).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(X.transpose(-2,-1)@self.EXTinvUX()@X).squeeze(-1).squeeze(-1)
        ELL = ELL + 0.5*self.ElogdetinvSigma() - 0.5*self.n*np.log(2*np.pi)
        for i in range(self.event_dim-2):
            ELL = ELL.sum(-1)
        return ELL

    def Elog_like_given_pX_pY(self,pX,pY):  # This assumes that X is a distribution with the ability to produce 
                                       # expectations of the form EXXT, and EX with dimensions matching Y, i.e. EX.shape[-2:] is (p,1)

        ELL = -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1)
        if self.pad_X:        
            ELL = ELL + (pY.mean().transpose(-2,-1)@(self.EinvUX()[...,:,:-1]@pX.mean()+self.EinvUX()[...,:,-1:])).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(pX.EXXT()*self.EXTinvUX()[...,:-1,:-1]).sum(-1).sum(-1)
            ELL = ELL - (self.EXTinvUX()[...,-1:,:-1]@pX.mean()).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(self.EXTinvUX()[...,-1,-1])            
        else:
            ELL = ELL + (pY.mean().transpose(-2,-1)@self.EinvUX()@pX.mean()).squeeze(-1).squeeze(-1)
            ELL = ELL - 0.5*(pX.EXXT()*self.EXTinvUX()).sum(-1).sum(-1)
        ELL = ELL + 0.5*self.invU.ElogdetinvSigma() - 0.5*self.n*np.log(2.0*np.pi)
        return ELL

    def Elog_like_X(self,Y):
        if self.pad_X:
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1]
            invSigmamu_x = self.EXTinvU()[...,:-1,:]@Y - self.EXTinvUX()[...,:-1,-1:]   #DOUBLECHECK HERE
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
            Residual = Residual - 0.5*self.EXTinvUX()[...,-1,-1]
        else:
            invSigma_x_x = self.EXTinvUX()
            invSigmamu_x = self.EXTinvU()@Y
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
        return invSigma_x_x, invSigmamu_x, Residual

    def Elog_like_X_given_pY(self,pY):
        if self.pad_X:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1]
            PmuJ_y = pY.EinvSigmamu() - self.EinvUX()[...,:,-1:]
            PmuJ_x = -self.EXTinvUX()[...,:-1,-1:]
            PJ_1_1 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX()
            PmuJ_y = pY.EinvSigmamu()
            PmuJ_x = torch.zeros(self.p,1)
            PJ_1_1 = torch.tensor(0)

        invSigma_y_y, negBinvD, negCinvA, invSigma_x_x = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        invSigmamu_x = PmuJ_x + negCinvA@PmuJ_y

        Sigma_x_x = invSigma_x_x.inverse()
        mu_x = Sigma_x_x@invSigmamu_x

        Res = pY.Res() + 0.5*(invSigmamu_y.transpose(-2,-1)@invSigma_y_y.inverse()@invSigmamu_y).squeeze(-1).squeeze(-1)
        Res = Res - 0.5*invSigma_y_y.logdet() + 0.5*pY.dim*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma() - 0.5*PJ_1_1
        return invSigma_x_x, invSigmamu_x, Res + 0.5*(mu_x*invSigmamu_x).sum(-1).sum(-1) - 0.5*invSigma_x_x.logdet() + 0.5*np.log(2*np.pi)*invSigmamu_x.shape[-2]

    def backward(self,pY):  
        if self.pad_X:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()[...,:,:-1]
            PJ_x_x = self.EXTinvUX()[...,:-1,:-1]
            PmuJ_y = pY.EinvSigmamu() + self.EinvUX()[...,:,-1:]
            PmuJ_x = -self.EXTinvUX()[...,:-1,-1:]
            PJ11 = self.EXTinvUX()[...,-1,-1]
        else:
            PJ_y_y = pY.EinvSigma() + self.EinvSigma()
            PJ_y_x = -self.EinvUX()
            PJ_x_x = self.EXTinvUX()
            PmuJ_y = pY.EinvSigmamu()
            PmuJ_x = torch.zeros(PJ_x_x.shape[:-1] + (1,))
            PJ11 = torch.tensor(0.0)

        invSigma_y_y, negBinvD, negCinvA, invSigma_x_x = matrix_utils.block_precision_marginalizer(PJ_y_y, PJ_y_x, PJ_y_x.transpose(-2,-1), PJ_x_x)
        invSigmamu_y = PmuJ_y + negBinvD@PmuJ_x
        invSigmamu_x = PmuJ_x + negCinvA@PmuJ_y

        px = MultivariateNormal_vector_format(invSigma = invSigma_x_x, invSigmamu = invSigmamu_x)
        Res = pY.Res() + 0.5*(invSigmamu_y.transpose(-2,-1)@invSigma_y_y.inverse()@invSigmamu_y).squeeze(-1).squeeze(-1) - 0.5*invSigma_y_y.logdet() + 0.5*pY.dim*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma() - 0.5*PJ11
        return px, Res - px.Res()

    def predict(self,X):
        if self.pad_X:
            invSigmamu_y = (self.EinvUX()[...,:,:-1]@X + self.EinvUX()[..., :,-1:])
        else:
            invSigmamu_y = (self.EinvUX()@X)
        invSigma_y_y = self.EinvSigma()
        Sigma_y_y = invSigma_y_y.inverse()
        mu_y = (Sigma_y_y@invSigmamu_y)

        return mu_y, Sigma_y_y, invSigma_y_y, invSigmamu_y

    def predict_given_pX(self,pX):
        return self.forward(pX)

    def mean(self):
        return self.mu

    def weights(self): 
        if self.pad_X is True:
            return self.mu[...,:-1]
        else:
            return self.mu

    def var(self):
        return self.ESigma().diagonal(dim1=-1,dim2=-2).unsqueeze(-1)*self.V.diagonal(dim1=-1,dim2=-2).unsqueeze(-2)

    ### Compute special expectations used for VB inference
    def EinvUX(self):
        device = self.mu.device
        
        # Get gamma mean and ensure it's on the correct device
        gamma_mean = self.invU.gamma.mean()
        if gamma_mean.device != device:
            gamma_mean = gamma_mean.to(device)
            
        return gamma_mean.unsqueeze(-1)*self.mu

    def EXTinvU(self):
        device = self.mu.device
        
        # Get EinvSigma and ensure it's on the correct device
        invU_EinvSigma = self.invU.EinvSigma()
        if invU_EinvSigma.device != device:
            invU_EinvSigma = invU_EinvSigma.to(device)
            
        return self.mu.transpose(-2,-1)@invU_EinvSigma

    def EXTAX(self, A):  # X is n x p, A is n x n
        device = self.mu.device
        
        # Ensure A is on the correct device
        if A.device != device:
            A = A.to(device)
            
        # Ensure V is on the correct device
        if self.V.device != device:
            V = self.V.to(device)
        else:
            V = self.V
            
        # Get meaninv and ensure it's on the correct device
        gamma_meaninv = self.invU.gamma.meaninv()
        if gamma_meaninv.device != device:
            gamma_meaninv = gamma_meaninv.to(device)
        
        # Ensure mu and A have compatible shapes
        if self.mu.shape[-2] != A.shape[-1]:
            # If mu has more rows than A has columns, truncate mu
            if self.mu.shape[-2] > A.shape[-1]:
                mu_adjusted = self.mu[:A.shape[-1], :]
            # If A has more columns than mu has rows, pad mu with zeros
            else:
                pad_size = A.shape[-1] - self.mu.shape[-2]
                mu_adjusted = torch.cat((self.mu, torch.zeros((pad_size,) + self.mu.shape[-1:], 
                                                            device=device, 
                                                            requires_grad=False)), dim=-2)
        else:
            mu_adjusted = self.mu
            
        return V*(gamma_meaninv*A.diagonal(dim1=-2,dim2=-1)).sum(-1) + mu_adjusted.transpose(-2,-1)@A@mu_adjusted

    def EXAXT(self, A):
        device = self.mu.device
        
        # Ensure A is on the correct device
        if A.device != device:
            A = A.to(device)
            
        # Ensure V is on the correct device
        if self.V.device != device:
            V = self.V.to(device)
        else:
            V = self.V
            
        # Get ESigma and ensure it's on the correct device
        ESigma = self.ESigma()
        if ESigma.device != device:
            ESigma = ESigma.to(device)
        
        # Ensure mu and A have compatible shapes
        if self.mu.shape[-2] != A.shape[-2]:
            # If mu has more rows than A has rows, truncate mu
            if self.mu.shape[-2] > A.shape[-2]:
                mu_adjusted = self.mu[:A.shape[-2], :]
            # If A has more rows than mu has rows, pad mu with zeros
            else:
                pad_size = A.shape[-2] - self.mu.shape[-2]
                mu_adjusted = torch.cat((self.mu, torch.zeros((pad_size,) + self.mu.shape[-1:], 
                                                            device=device, 
                                                            requires_grad=False)), dim=-2)
        else:
            mu_adjusted = self.mu
            
        return ESigma*(V*A).sum(-1).sum(-1) + mu_adjusted@A@mu_adjusted.transpose(-2,-1)

    def EXTinvUX(self):
        # Ensure mu and V have compatible shapes
        device = self.mu.device
        
        if self.mu.shape[-1] != self.V.shape[-1]:
            # If mu has more columns than V, truncate mu
            if self.mu.shape[-1] > self.V.shape[-1]:
                mu_adjusted = self.mu[..., :self.V.shape[-1]]
            # If V has more columns than mu, pad mu with zeros
            else:
                pad_size = self.V.shape[-1] - self.mu.shape[-1]
                mu_adjusted = torch.cat((self.mu, torch.zeros(self.mu.shape[:-1] + (pad_size,), 
                                                            device=device, 
                                                            requires_grad=False)), dim=-1)
        else:
            mu_adjusted = self.mu
        
        # Get gamma mean and ensure it's on the correct device
        gamma_mean = self.invU.gamma.mean()
        if gamma_mean.device != device:
            gamma_mean = gamma_mean.to(device)
            
        # Ensure V is on the correct device
        if self.V.device != device:
            V = self.V.to(device)
        else:
            V = self.V
            
        # Ensure n is on the correct device
        n = self.n
        if isinstance(n, torch.Tensor) and n.device != device:
            n = n.to(device)
            
        return n * V + mu_adjusted.transpose(-1,-2)@(gamma_mean.unsqueeze(-1)*mu_adjusted)

    def EXinvVXT(self):
        # Ensure mu and invV have compatible shapes
        device = self.mu.device
        
        if self.mu.shape[-1] != self.invV.shape[-1]:
            # If mu has more columns than invV, truncate mu
            if self.mu.shape[-1] > self.invV.shape[-1]:
                mu_adjusted = self.mu[..., :self.invV.shape[-1]]
            # If invV has more columns than mu, pad mu with zeros
            else:
                pad_size = self.invV.shape[-1] - self.mu.shape[-1]
                mu_adjusted = torch.cat((self.mu, torch.zeros(self.mu.shape[:-1] + (pad_size,), 
                                                            device=device, 
                                                            requires_grad=False)), dim=-1)
        else:
            mu_adjusted = self.mu
            
        # Ensure invV is on the correct device
        if self.invV.device != device:
            invV = self.invV.to(device)
        else:
            invV = self.invV
            
        # Get ESigma and ensure it's on the correct device
        invU_ESigma = self.invU.ESigma()
        if invU_ESigma.device != device:
            invU_ESigma = invU_ESigma.to(device)
            
        # Ensure p is on the correct device
        p = self.p
        if isinstance(p, torch.Tensor) and p.device != device:
            p = p.to(device)
            
        return p * invU_ESigma + mu_adjusted@invV@mu_adjusted.transpose(-1,-2)

    def EXmMUTinvUXmMU(self): # X minus mu
        device = self.mu.device
        
        # Get n and ensure it's on the correct device
        n = self.n
        if isinstance(n, torch.Tensor) and n.device != device:
            n = n.to(device)
            
        # Ensure V is on the correct device
        if self.V.device != device:
            V = self.V.to(device)
        else:
            V = self.V
            
        return n*V

    def EXmMUinvVXmMUT(self):
        device = self.mu.device
        
        # Get p and ensure it's on the correct device
        p = self.p
        if isinstance(p, torch.Tensor) and p.device != device:
            p = p.to(device)
            
        # Get ESigma and ensure it's on the correct device
        invU_ESigma = self.invU.ESigma()
        if invU_ESigma.device != device:
            invU_ESigma = invU_ESigma.to(device)
            
        return p*invU_ESigma

    def EXTX(self):
        # Ensure mu has compatible shape with itself for the matrix multiplication
        device = self.mu.device
        meaninv = self.invU.gamma.meaninv()
        if meaninv.device != device:
            meaninv = meaninv.to(device)
        
        if self.V.device != device:
            V = self.V.to(device)
        else:
            V = self.V
            
        return V * meaninv.sum() + self.mu.transpose(-1,-2)@self.mu

    def EXXT(self):
        device = self.mu.device
        V_diag = self.V.diagonal()
        if V_diag.device != device:
            V_diag = V_diag.to(device)
            
        invU_ESigma = self.invU.ESigma()
        if invU_ESigma.device != device:
            invU_ESigma = invU_ESigma.to(device)
            
        return V_diag.sum() * invU_ESigma + self.mu@self.mu.transpose(-1,-2)

    def ElogdetinvU(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def ElogdetinvSigma(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def EinvSigma(self):  
        return self.invU.mean()

    def ESigma(self):  
        return self.invU.ESigma()


class MatrixNormalGamma_UnitTrace(MatrixNormalGamma):

    def __init__(self,mu_0,U_0=None,V_0=None,mask=None,X_mask=None,pad_X=False):
        super().__init__(mu_0,U_0=U_0,V_0=V_0,uniform_precision=False,mask=mask,X_mask=X_mask,pad_X=pad_X)
        alpha = self.invU.gamma.alpha_0
        beta = self.invU.gamma.beta_0
        self.invU = DiagonalWishart_UnitTrace(alpha,beta)

#    def ss_update(self,SExx,SEyx,SEyy,n,lr=1.0):
#        super().ss_update(SExx,SEyx,SEyy,n,lr=lr)
#        self.mu = self.mu*self.invU.rescale.unsqueeze(-1).sqrt()


