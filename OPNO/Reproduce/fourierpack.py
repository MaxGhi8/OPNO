import torch

def sin_transform(u):
    Nx = u.shape[-1]
    V = torch.cat([u, -u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = -torch.fft.fft(V, dim=-1)[..., :Nx].imag# / Nx
    return a

def isin_transform(a):
    Nx = a.shape[-1]
    V = torch.cat([a, -a.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].imag# * Nx
    return u

def cos_transform(u):
    Nx = u.shape[-1]

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.fft(V, dim=-1)[..., :Nx].real# / Nx
    return a

def icos_transform(a):
    Nx = a.shape[-1]

    V = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].real# * Nx
    return u

def dctII(u):
    Nx = u.shape[-1]

    v = torch.cat([u[..., ::2], u[..., 1::2].flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(Nx, dtype=u.dtype, device=u.device)
    W4 = torch.exp(-.5j * torch.pi * k / Nx)
    return 2 * (V * W4).real / Nx

def idctII(a):
    Nx = a.shape[-1]

    k = torch.arange(Nx, dtype=a.dtype, device=a.device)
    iW4 = 1 / torch.exp(-.5j * torch.pi * k / Nx); iW4[..., 0] /= 2

    V = torch.fft.ifft(a * iW4).real
    u = torch.zeros_like(V)
    u[..., ::2], u[..., 1::2] = V[..., :Nx - (Nx // 2)], V.flip(dims=[-1])[..., :Nx // 2]

    return u * Nx

def dstII(u):
    v = u.clone()
    v[..., 1::2] = -v[..., 1::2]
    return dctII(v).flip(dims=[-1])

def idstII(a):
    v = idctII(a.flip(dims=[-1]))
    u = v.clone()
    u[..., 1::2] = -u[..., 1::2]
    return u

def fourier_partial(u, d=-1, T=sin_transform, iT=icos_transform):
    total_dim = u.dim()
    if (d != total_dim-1) and (d != -1):
        u = torch.transpose(u, d, -1)

    Nx = u.shape[-1]
    k = (torch.linspace(0, Nx-1, Nx))
    du_fft = iT(T(u) * k).real

    if (d != total_dim-1) and (d != -1):
        du_fft = torch.transpose(du_fft, d, -1)
    return du_fft

def fourier_partial2(u, T, iT):
    Nx = u.shape[-1]
    k = -(torch.linspace(1, Nx, Nx)) ** 2
    du_fft = iT(T(u) * k).real
    return du_fft
