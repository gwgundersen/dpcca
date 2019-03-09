"""=============================================================================
Test that our implementation of a vectorized outer product is correct.
============================================================================="""

import torch
import unittest

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def test_einsum(self):
        k = 2
        n = 5
        X = torch.randn(k, n)
        XX1 = torch.einsum('ab,cb->ac', [X, X])
        XX2 = torch.zeros(k, k)
        for i in range(n):
            x = X[:, i]
            XX2 += torch.ger(x, x)
        self.assertTrue(torch.allclose(XX1, XX2))

    def test_einsum_differentiable(self):
        # See here for why this test exists:
        #
        #     https://github.com/pytorch/pytorch/issues/7763
        #
        x = torch.randn(3, 3, requires_grad=True)
        z = torch.einsum("ij,jk->ik", (x, torch.randn(3, 3)))
        try:
            z.sum().backward()
        except RuntimeError:
            self.fail('einsum() raised RuntimeError unexpectedly.')
