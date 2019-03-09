"""=============================================================================
Functions for linear algebra operations.
============================================================================="""

import cuda
import torch

# ------------------------------------------------------------------------------

diag = torch.diag
inv  = torch.inverse

# ------------------------------------------------------------------------------

def to_positive(A_diag, eps=0.00001):
    """Convert n-vector into an n-vector with nonnegative entries.
    """
    A_diag[A_diag < 0] = eps
    inds = torch.isclose(A_diag, torch.zeros(1, device=cuda.device()))
    A_diag[inds] = eps
    return A_diag

# ------------------------------------------------------------------------------

def woodbury_inv(A_diag, U, V, k):
    """This matrix inversion is O(k^3) rather than O(p^3) where p is the
    dimensionality of the data and k is the latent dimension.
    """
    # Helps with numerics. If A_diag[i, j] == 0, then 1 / 0 == inf.
    SMALL = 1e-12
    A_inv_diag = 1. / (A_diag + SMALL)

    I     = torch.eye(k, device=cuda.device())
    B_inv = inv(I + ((V * A_inv_diag) @ U))

    # We want to perform the operation `U @ B_inv @ V` but need to optimize it:
    # - Computing `tmp1` is fast because it is (p, k) * (k, k).
    # - Computing `tmp2` is slow because it is (p, k) * (k, p).
    tmp1  = U @ B_inv
    tmp2  = torch.einsum('ab,bc->ac', (tmp1, V))

    # Use `view` rather than `reshape`. The former guarantees that a new tensor
    # is returned.
    tmp3  = A_inv_diag.view(-1, 1) * tmp2
    right = tmp3 * A_inv_diag

    # This is a fast version of `diag(A_inv_diag) - right`.
    right = -1 * right
    idx   = torch.arange(0, A_diag.size(0), device=cuda.device())
    right[idx, idx] = A_inv_diag + right[idx, idx]

    return right

# ------------------------------------------------------------------------------

def diag_inv(A):
    """The inverse of a diagonal matrix is just the reciprocal of each of its
    diagonal elements
    """
    return diag(1. / diag(A))

# ------------------------------------------------------------------------------

def sum_outers(x, y):
    """Return sum of outer products of paired columns of x and y.
    """
    # In PyTorch 4.0, `einsum` modifies variables inplace. This will not work
    # unless you have PyTorch 4.1:
    #
    #     https://github.com/pytorch/pytorch/issues/7763
    #
    return torch.einsum('ab,cb->ac', [x, y])
