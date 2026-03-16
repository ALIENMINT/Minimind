import torch.nn as nn
import torch

# derived nn.Module
class RMSNorm(nn.Module):
    def __init__ (self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return x * self._norm(x.float()).type_as(x) * self.weight