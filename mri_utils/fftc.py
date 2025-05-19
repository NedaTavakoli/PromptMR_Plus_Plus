"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
import torch.fft

def fft2c_new(data, norm='ortho'):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    
    Args:
        data: Complex valued input data with at least 3 dimensions. The last dimension should
            be 2 (real and imaginary parts).
        norm: Normalization method.
        
    Returns:
        The FFT of the input.
    """
    if torch.is_complex(data):
        # Data is already complex, convert to real tensor with last dim of 2
        data = torch.stack([data.real, data.imag], dim=-1).to(torch.float32)
    
    if data.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")
    
    # Make sure data is float32
    if data.dtype != torch.float32:
        data = data.to(torch.float32)
    
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.fft2(data, dim=(-2, -1), norm=norm)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    
    return data

def ifft2c_new(data, norm='ortho'):
    """
    Apply centered 2 dimensional Inverse Fast Fourier Transform.
    
    Args:
        data: Complex valued input data with at least 3 dimensions. The last dimension should
            be 2 (real and imaginary parts).
        norm: Normalization method.
        
    Returns:
        The IFFT of the input.
    """
    if torch.is_complex(data):
        # Data is already complex, convert to real tensor with last dim of 2
        data = torch.stack([data.real, data.imag], dim=-1).to(torch.float32)
    
    if data.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")
    
    # Make sure data is float32
    if data.dtype != torch.float32:
        data = data.to(torch.float32)
    
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifft2(data, dim=(-2, -1), norm=norm)
    data = torch.view_as_real(data)
    data = fftshift(data, dim=(-3, -2))
    
    return data

# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    
    for d in dim:
        n_shift = x.size(d) // 2
        if x.size(d) % 2 != 0:
            n_shift = (x.size(d) - 1) // 2
        x = torch.roll(x, shifts=n_shift, dims=d)
    
    return x

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    
    for d in dim:
        n_shift = (x.size(d) + 1) // 2
        if x.size(d) % 2 != 0:
            n_shift = ((x.size(d) + 1) // 2) - 1
        x = torch.roll(x, shifts=-n_shift, dims=d)
    
    return x

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)