"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os

from utilities import *

import fourierpack as sp
import chebypack as ch
from functools import partial as PARTIAL


class Transform:
    def __init__(self, fwd, inv):
        assert (type(fwd)== functools.partial and type(inv) == functools.partial)
        self.fwd = fwd
        self.inv = inv
    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)

class PseudoSpectra(nn.Module):
    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super(PseudoSpectra, self).__init__()

        self.T = T
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)

        # print([(l, 0) for l in triL])
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels*bandwidth.prod().item(), out_channels, modes.prod().item()))
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
                                      padding=triL)
        self.X_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes+bandwidth-1-triL*2]

    def quasi_diag_mul(self, input, weights):
        xpad = self.unfold(input)
        # print(xpad.shape, weights.shape)
        return torch.einsum("bix, iox->box", xpad, weights)

    def forward(self, u):
        batch_size = u.shape[0]

        b = self.T(u, self.X_dims)

        out = torch.zeros(batch_size, self.out_channels, *u.shape[2:], device=u.device, dtype=u.dtype)
        out[self.X_slices] = self.quasi_diag_mul(b[self.pad_slices], self.weights).reshape(
                batch_size, self.out_channels, *self.modes)

        u = self.T.inv(out, self.X_dims)
        return u

class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(1, device = x.device)

class SOL(nn.Module):
    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL = 0):
        super(SOL, self).__init__()

        modes = np.array([modes]*dim) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]*dim) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]*dim) if isinstance(triL, int) else np.array(triL)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.conv0 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)

        self.lift = PseudoSpectra(T, dim, in_channels, width-in_channels, modes, bandwidth, triL)

        self.w0 = convND(width, width, 1)
        self.w1 = convND(width, width, 1)
        self.w2 = convND(width, width, 1)
        self.w3 = convND(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()

    def forward(self, x):

        # [batch, XYZ, c] -> [batch, c, XYZ]
        x = x.permute(0, -1, *self.X_dims-1)

        x = torch.cat([x, F.gelu(self.lift(x))], dim=1)

        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x))

        x = x.permute(0, *self.X_dims, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # x = self.T.inv(self.T(x, self.X_dims-1), self.X_dims-1)

        return x

def Wrapper(func_list, u, dim):
    # a wrapper to apply a list of function on given axises.
    # the func will be applied in turn.
    if type(dim) == int:
        dim = [dim]
    total_dim = u.dim()

    for d in dim:
        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)

        for func in func_list:
            u = func(u)

        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u

def FuncMat_Wrapper(func_mat, u, dim):
    total_dim = u.dim()

    for d, func_list in zip(dim, func_mat):
        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)

        for func in func_list:
            u = func(u)

        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u

_dst = PARTIAL(Wrapper, [sp.sin_transform])
_idst = PARTIAL(Wrapper, [sp.isin_transform])
_dct = PARTIAL(Wrapper, [sp.cos_transform])
_idct = PARTIAL(Wrapper, [sp.icos_transform])
_dcht = PARTIAL(Wrapper, [ch.dct])
_idcht = PARTIAL(Wrapper, [ch.idct])

_idctII = PARTIAL(Wrapper, [sp.idctII])
_dctII = PARTIAL(Wrapper, [sp.dctII])
_idstII = PARTIAL(Wrapper, [sp.idstII])
_dstII = PARTIAL(Wrapper, [sp.dstII])

_dShenT_Dirichlet = PARTIAL(Wrapper, [ch.dct, ch.cmp])
_dShenT_Neumann = PARTIAL(Wrapper, [ch.dct, ch.cmp_neumann])
_dShenT_Robin = PARTIAL(Wrapper, [ch.dct, ch.cmp_robin])
_idShenT_Dirichlet = PARTIAL(Wrapper, [ch.icmp, ch.idct])
_idShenT_Neumann = PARTIAL(Wrapper, [ch.icmp_neumann, ch.idct])
_idShenT_Robin = PARTIAL(Wrapper, [ch.icmp_robin, ch.idct])

#### Transform

## Transform for SPFNO

DST = Transform(_dst, _idst)
DCT = Transform(_dct, _idct)
DST_II = Transform(_dstII, _idstII)
DCT_II = Transform(_dctII, _idctII)

## Transform for OPNO
Chebyshev_Shen_Transform_Dirichlet = Transform(_dcht, _idShenT_Dirichlet)
Chebyshev_Shen_Transform_Neumann = Transform(_dcht, _idShenT_Neumann)
Chebyshev_Shen_Transform_Robin = Transform(_dcht, _idShenT_Robin)
full_Chebyshev_Shen_Transform_Dirichlet = Transform(_dShenT_Dirichlet, _idShenT_Dirichlet)
full_Chebyshev_Shen_Transform_Neumann = Transform(_dShenT_Neumann, _idShenT_Neumann)
full_Chebyshev_Shen_Transform_Robin = Transform(_dShenT_Robin, _idShenT_Robin)


#### Neural Operators

SinNO = PARTIAL(SOL, DST)
CosNO = PARTIAL(SOL, DCT)
SinNO_II = PARTIAL(SOL, DST_II)
CosNO_II = PARTIAL(SOL, DCT_II)

SinNO1d = PARTIAL(SOL1d, DST)
CosNO1d = PARTIAL(SOL1d, DCT)
SinNO2d = PARTIAL(SOL, DST, dim=2)
CosNO2d = PARTIAL(SOL, DCT, dim=2)

OPNO_Neumann2d = PARTIAL(SOL, Chebyshev_Shen_Transform_Neumann, dim=2)
