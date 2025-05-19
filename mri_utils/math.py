"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()

import torch

def complex_center_crop(data, shape):
    """
    Apply center crop to complex-valued tensor.
    
    Args:
        data: Input complex data with last dimension being 2 (real and imaginary)
        shape: Desired output shape
        
    Returns:
        Cropped data
    """
    if not isinstance(shape, (list, tuple)):
        shape = (shape, shape)
    
    if all(x == y for x, y in zip(data.shape[-3:-1], shape)):
        return data
    
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-3] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    if len(data.shape) == 3:
        return data[h_from:h_to, w_from:w_to, :]
    elif len(data.shape) == 4:
        return data[:, h_from:h_to, w_from:w_to, :]
    elif len(data.shape) == 5:
        return data[:, :, h_from:h_to, w_from:w_to, :]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

def complex_random_crop(data, shape):
    """
    Apply random crop to complex-valued tensor.
    
    Args:
        data: Input complex data with last dimension being 2 (real and imaginary)
        shape: Desired output shape
        
    Returns:
        Cropped data
    """
    if not isinstance(shape, (list, tuple)):
        shape = (shape, shape)
    
    if all(x == y for x, y in zip(data.shape[-3:-1], shape)):
        return data
    
    w_from = torch.randint(0, data.shape[-2] - shape[0] + 1, (1,)).item()
    h_from = torch.randint(0, data.shape[-3] - shape[1] + 1, (1,)).item()
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    if len(data.shape) == 3:
        return data[h_from:h_to, w_from:w_to, :]
    elif len(data.shape) == 4:
        return data[:, h_from:h_to, w_from:w_to, :]
    elif len(data.shape) == 5:
        return data[:, :, h_from:h_to, w_from:w_to, :]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")