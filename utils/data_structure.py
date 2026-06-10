# -*- coding:utf-8 -*-
import torch
import numpy as np


def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), allow_unused=True, create_graph=True)


def get_param_num(model):
    return sum(p.numel() for p in model.parameters())


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def to_tensor(input, dtype=torch.float32, requires_grad=False, device=None):
    if device is None:
        return torch.tensor(input, dtype=dtype, requires_grad=requires_grad)
    else:
        return torch.tensor(input, dtype=dtype, requires_grad=requires_grad).to(device)


def remove_mask(*arrays, mask, axis=None):
    """
    Removes the elements at the specified indices (mask) from each input array.
    
    Parameters:
    *arrays : array-like
        A variable number of input arrays.
    mask : list, tuple, or numpy.ndarray
        The indices of the elements to remove.
    axis : int, optional
        The axis along which to delete the subarray defined by mask. 
        If None, the array is flattened first (default numpy behavior).
        
    Returns:
    tuple of numpy.ndarray
        A tuple containing the arrays with the masked elements removed.
    """
    # Handle edge cases where mask is None or empty to avoid numpy errors
    if mask is None or (hasattr(mask, '__len__') and len(mask) == 0):
        return tuple(np.asarray(arr) for arr in arrays)
        
    return tuple(np.delete(arr, mask, axis=axis) for arr in arrays)