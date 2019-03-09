"""=============================================================================
Verify matrix inversion of diagonal matrices is both fast and accurate.
============================================================================="""

import unittest

import time
import torch
import linalg

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def test_diag_inv_accuracy(self):
        A = torch.eye(20) * torch.randn(20)
        A_diag = linalg.diag_inv(A)
        self.assertTrue(torch.allclose(A @ A_diag, torch.eye(len(A)),
                                       atol=0.01))

    def test_diag_inv_speed(self):
        A = torch.eye(1000) * torch.randn(1000)

        s1 = time.time()
        torch.inverse(A)
        d1 = time.time() - s1

        s2 = time.time()
        linalg.diag_inv(A)
        d2 = time.time() - s2

        # According to my tests, d1 is roughly 2 orders of magnitude slower.
        self.assertTrue(d1 > d2)

