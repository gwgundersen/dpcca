"""=============================================================================
Verify PyTorch's slice functionality is differentiable as we expect.
============================================================================="""

import unittest
import torch

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def test_without_slice(self):

        x = torch.ones(4) * 2  # [2, 2, 2, 2]
        x.requires_grad_(True)
        y = x**3  # [8, 8]
        y.backward(torch.ones(4))

        # y  = x^3
        # y' = 3x^2
        # y' @ 2 = 12
        self.assertTrue((x.grad == torch.Tensor([12, 12, 12, 12])).all())

    def test_slice(self):

        x = torch.ones(4) * 2  # [2, 2, 2, 2]
        x.requires_grad_(True)

        x1 = x[:2]  # [2, 2]
        x2 = x[2:]  # [2, 2]

        y1 = x1**2  # [4, 4]
        y2 = x2**3  # [8, 8]

        y = torch.cat([y1, y2])  # [4, 4, 8, 8]
        y.backward(torch.ones(4))

        # y  = x^2      y  = x^3
        # y' = 2x       y' = 3x^2
        # y' @ 2 = 4    y' @ 2 = 12
        self.assertTrue((x.grad == torch.Tensor([4, 4, 12, 12])).all())
