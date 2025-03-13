import torch
class Delta():
    def __init__(self,X):
        self.X = X

    def unsqueeze(self,dim):  # only appliles to batch
        self.X = self.X.unsqueeze(dim)
        return self

    def squeeze(self,dim):  # only appliles to batch
        self.X = self.X.squeeze(dim)
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
        if hasattr(self, 'X') and self.X is not None:
            self.X = self.X.to(device)
            
        return self

    # def Elog_like(self):
    #     torch.ones(self.X.shape[:-self.event_dim],requires_grad=False)

    # def KLqprior(self):
    #     return torch.zeros(self.X.shape[:-self.event_dim],requires_grad=False)

    # def ELBO(self):
    #     return torch.zeros(self.X.shape[:-self.event_dim],requires_grad=False)
    @property
    def shape(self):
        return self.X.shape
        
    @property
    def mu(self):
        """Return X as mu for compatibility with other distribution classes"""
        return self.X

    def mean(self):
        return self.X

    def EX(self):
        return self.X

    def EXXT(self):
        return self.X@self.X.transpose(-1,-2)

    def EXTX(self):
        return self.X.transpose(-1,-2)@self.X

    def EXTAX(self,A):
        return self.X.transpose(-1,-2)@A@self.X

    def EXX(self):
        return self.X**2

    def ElogX(self):
        return torch.log(self.X)

    def E(self,f):
        return f(self.X)

    def logZ(self):
        return torch.zeros(self.batch_shape)


