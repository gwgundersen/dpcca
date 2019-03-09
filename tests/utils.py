"""=============================================================================
Utility functions for unit testing.
============================================================================="""

import numpy as np
import torch

# ------------------------------------------------------------------------------

def relaxed_allclose(x, y, atol=0.01, pct_wrong_allowable=0.1):
    """Comparing two Torch tensors using `allclose` repeatedly fails due to
    numerical issues. This test relaxes the constraint that every single element
    in the two tensors are similar. Instead, a percentage of elements must be
    similar.
    """
    res     = torch.isclose(x, y, atol=atol)
    n_wrong = res.numel() - res.sum()
    n_wrong_allowable = pct_wrong_allowable * res.numel()
    return n_wrong <= n_wrong_allowable

# ------------------------------------------------------------------------------

def gen_simple_dataset(p1, p2, k, n, sigma1, sigma2):
    """Generate a paired dataset (y1, y2) with a known latent variable z.

    :param p1:     Dimension of dataset y1.
    :param p2:     Dimension of dataset y2.
    :param k:      Dimension of latent variable z.
    :param n:      Number of samples.
    :param sigma1: Variance of y1.
    :param sigma2: Variance of y2.
    :return:
    """
    y1 = np.zeros((p1, n))
    y2 = np.zeros((p2, n))

    Lambda1 = np.random.random((p1, k))
    Lambda2 = np.random.random((p2, k))
    z = np.random.random((k, n))

    m1 = np.dot(Lambda1, z)
    m2 = np.dot(Lambda2, z)

    Psi1 = sigma1 * np.eye(p1)
    Psi2 = sigma2 * np.eye(p2)

    for i in range(n):
        y1[:, i] = np.random.multivariate_normal(mean=m1[:, i], cov=Psi1)
        y2[:, i] = np.random.multivariate_normal(mean=m2[:, i], cov=Psi2)

    # For historical reasons, the input shape of the data is always
    # (n_samples, n_features)
    return torch.Tensor(y1).t(), torch.Tensor(y2).t(), torch.Tensor(z)
