import torch
import numpy as np

def CGL_points(Nx:int, dtype=torch.float32) -> torch.Tensor:
    return torch.cos(torch.linspace(0, torch.pi, Nx, dtype=dtype))

def dct(u):
    Nx = u.shape[-1]

    # transform x -> theta, a discrete cosine transform of "cheap" version
    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2
    return a

def idct(a):
    Nx = a.shape[-1]

    v = a.clone()
    v[..., (0, Nx-1)] *= 2
    V = torch.cat([v, v.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real / 2
    return u

def cmp(a):
    Nx = a.shape[-1]

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1.0
    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
    # b[..., -4:-2] = -a[..., -2:]
    # print(b[-2:])
    # b[..., -2:] = 0
    return b


def galerkin(a):
    # c_k = [2, 1, 1, 1....]
    Nx = a.shape[-1]

    b = torch.zeros_like(a)
    b[..., :-2] = torch.pi/2 * (a[..., :-2] - a[..., 2:])
    b[..., 0] += torch.pi/2 * a[..., 0]
    return b


def cmp_decrease(a):
    Nx = a.shape[-1]

    sgn = torch.zeros(*a.shape[:-1], 2*Nx, dtype=a.dtype, device=a.device)
    sgn[..., -(Nx-1)//2*2::2] = -1.0

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
    b[..., -2:] = 0

    return b

def cmp_neumann(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device) ** 2

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    b[..., -2:] = 0

    return b

def cmp_robin(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device)
    fac = (fac**2+1)
    # fac = (fac-1.0)*(fac+1.0)

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    return b

def icmp(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)
    a[..., :2] = b[..., :2]
    a[..., 2:] = b[..., 2:] - b[..., :Nx-2]
    a[..., -2:] = -b[..., Nx-4:Nx-2]

    return a

def icmp_neumann(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    p = torch.linspace(0, Nx-3, Nx-2, dtype=b.dtype, device=b.device)
    p = ( (p/(p+2.0))**2)
    a[..., 0:2] = b[..., 0:2]
    a[..., 2:Nx-2] = b[..., 2:Nx-2] - p[:Nx-4] * b[..., :Nx-4]
    a[..., Nx-2:Nx] = -p[Nx-4:Nx-2] * b[..., Nx-4:Nx-2]

    return a

def icmp_robin(b):
    # pk = (p * k^2 +1) / (p * (k+2)^2 + 1)
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    pk = torch.linspace(0, Nx-3, Nx-2, dtype=b.dtype, device=b.device)
    pk = (pk**2+1) / ((pk+2.0)**2+1)

    a[..., 0:2] = b[..., 0:2]
    a[..., 2:Nx-2] = b[..., 2:Nx-2] - pk[:Nx-4] * b[..., :Nx-4]
    a[..., Nx-2:Nx] = -pk[Nx-4:Nx-2] * b[..., Nx-4:Nx-2]

    return a

def cheb_partial(u, d, truc = None):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2

    a *= 2 * torch.linspace(0, Nx-1, Nx, dtype=u.dtype, device=u.device)
    sgn = torch.zeros(2*Nx, device=a.device, dtype=u.dtype)
    sgn[..., Nx//2*2+3::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    if truc != None:
        b[..., truc:] = 0

    b[..., 0] /= 2

    a = b

    a[..., 1:Nx-1] /= 2
    V = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real# / 2

    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)
    return u

Dx = cheb_partial

def Value_on_boundary(u, d, truc = None):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2

    a *= 2 * torch.linspace(0, Nx-1, Nx, dtype=u.dtype, device=u.device)

    b = torch.zeros_like(a)

    b[..., (Nx-3, Nx-2)] = a[..., (Nx-2, Nx-1)]
    for j in reversed(range(Nx-3)):
        b[..., j] = b[..., j+2] + a[..., j+1]

    print(b.sum(dim=-1))
    return b

def cmp_UpperDirichlet(a):
    b = a.cumsum(dim=-1)
    b[..., -2] = -a[..., -1]
    b[..., -1] = 0
    return b

def icmp_UpperDirichlet(b):
    a = torch.zeros_like(b)
    a[..., 1:-1] = b[..., 1:-1] - b[..., :-2]
    a[..., 0] = b[..., 0]
    a[..., -1] = -b[..., -2]
    return a
