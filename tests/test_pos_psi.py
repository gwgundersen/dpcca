"""=============================================================================
Test our function ensuring Psi is nonnegative works as expected.
============================================================================="""

import unittest
import torch
import linalg as LA

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def test_to_pos(self):
        for _ in range(1000):
            A_diag = LA.to_positive(torch.randn(10))
            self.assertTrue((A_diag > 0).all())

            try:
                torch.cholesky(torch.diag(A_diag))
            except RuntimeError:
                self.fail('Call to `torch.cholesky` failed.')
