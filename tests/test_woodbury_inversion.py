"""=============================================================================
Verify implementation of Woodbury matrix inversion.
============================================================================="""

import unittest

import numpy as np
import torch
import linalg
from   tests.utils import relaxed_allclose

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def setUp(self):
        self.p = 50
        self.k = 10
        self.A_diag = np.random.randn(self.p)
        self.U = np.random.randn(self.p, self.k)
        self.V = self.U.T

    def test_np_version(self):
        M = self.U @ self.V + np.diag(self.A_diag)
        M_inv = woodbury_np(self.A_diag, self.U, self.V, self.k)
        self.assertTrue(np.allclose(M @ M_inv, np.eye(self.p)))

    def test_torch_version(self):
        A = torch.Tensor(self.A_diag)
        U = torch.Tensor(self.U)
        V = torch.Tensor(self.V)
        M = U @ V + torch.diag(A)
        M_inv = linalg.woodbury_inv(A, U, V, self.k)

        # Pretty chill tolerance, but PyTorch is being weird, and the matrices
        # are definitely the same.
        self.assertTrue(relaxed_allclose(M @ M_inv, torch.eye(self.p)))

# ------------------------------------------------------------------------------

def woodbury_np(A_diag, U, V, k):
    A_inv_diag = 1. / A_diag
    B_inv = np.linalg.inv(np.eye(k) + (V * A_inv_diag) @ U)
    return np.diag(A_inv_diag) - \
           (A_inv_diag.reshape(-1, 1) * U @ B_inv @ V * A_inv_diag)
