import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Spectral(nn.Module):
    """Spectral layer base model
    Base Spectral layer model as presented in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.054312,
    implemented using PyTorch-based code. Here eigenvectors and eigenvalues are sampled as nn.Linear layer in PyTorch
    Parameters
    ----------
    in_dim:
        Input size
    out_dim:
        Output size
    base_grad:
        If set to True the eigenvectors are trainable
    start_grad:
        If set to True the starting eigenvalues are trainable
    end_grad:
        If set to True the ending eigenvalues are trainable
    bias:
        If set to True add bias
    device:
        Device for training
    dtype:
        Type for the training parameters
    Example
    -------
    model = torch.nn.Sequential(
                            Spectral(1, 20),
                            Spectral(20,20),
                            F.elu()
                            spectral(20,1)
                            )
    """

    __constants__ = ['in_dim', 'out_dim']
    in_dim: int
    out_dim: int

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 base_grad: bool = False,
                 start_grad: bool = True,
                 end_grad: bool = True,
                 bias: bool = False,
                 device=None,
                 dtype=torch.float,
     ):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Spectral, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_grad = base_grad
        self.start_grad = start_grad
        self.end_grad = end_grad

        # Build the model

        # Eigenvectors
        self.base = nn.Parameter(torch.empty(self.in_dim, self.out_dim, **factory_kwargs), requires_grad=self.base_grad)
        # Eigenvalues start
        self.diag_start = nn.Parameter(torch.empty(self.in_dim, 1, **factory_kwargs), requires_grad=self.start_grad)
        # Eigenvalues end
        self.diag_end = nn.Parameter(torch.empty(1, self.out_dim, **factory_kwargs), requires_grad=self.end_grad)
        # bias
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        # Initialize the layer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # Same of torch.nn.modules.linear
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base)
        bound_in = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bound_out = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
        nn.init.uniform_(self.diag_start, -bound_in, bound_in)
        nn.init.uniform_(self.diag_end, -bound_out, bound_out)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound_in, bound_in)

    def forward(self, x):
        kernel = torch.mul(self.base, self.diag_start - self.diag_end)
        if self.bias:
            outputs = F.linear(x, kernel.t(), self.bias)
        else:
            outputs = F.linear(x, kernel.t())

        return outputs

    def extra_repr(self) -> str:
        return 'in_dim={}, out_dim={}, base_grad={}, start_grad={}, end_grad={}, bias={}'.format(
            self.in_dim, self.out_dim, self.base_grad, self.start_grad, self.end_grad, self.bias is not None
        )
