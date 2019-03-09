"""=============================================================================
Test PCCA model end-to-end.
============================================================================="""

import torch
import unittest
from   models import PCCAVec, PCCASimple

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def setUp(self):
        n  = 1000
        p1 = 50
        p2 = 30
        k  = 10
        p  = p1 + p2

        # Initialize parameters to verify that both implementations compute the
        # same values.
        self.Lambda1   = torch.randn(p1, k)
        self.Lambda2   = torch.randn(p2, k)
        self.Psi1_diag = torch.randn(p1)
        self.Psi2_diag = torch.randn(p2)
        self.y = torch.randn(p, n)

        self.p1 = p1; self.p2 = p2; self.k = k

    def test_same_parameters_after_1_iters(self):
        n_iters = 1

        pccas = PCCASimple(latent_dim=self.k, dims=[self.p1, self.p2],
                                n_iters=n_iters)
        pccas.Lambda1.data   = self.Lambda1
        pccas.Lambda2.data   = self.Lambda2
        pccas.Psi1_diag.data = self.Psi1_diag
        pccas.Psi2_diag.data = self.Psi2_diag

        pccav = PCCAVec(latent_dim=self.k, dims=[self.p1, self.p2],
                             n_iters=n_iters)
        pccav.Lambda1.data   = self.Lambda1
        pccav.Lambda2.data   = self.Lambda2
        pccav.Psi1_diag.data = self.Psi1_diag
        pccav.Psi2_diag.data = self.Psi2_diag

        pccas.forward(self.y)
        pccav.forward(self.y)

        atol = 0.01
        self.assertTrue(torch.allclose(pccas.Lambda1, pccav.Lambda1, atol=atol))
        self.assertTrue(torch.allclose(pccas.Lambda2, pccav.Lambda2, atol=atol))
        self.assertTrue(torch.allclose(pccas.Psi1_diag, pccav.Psi1_diag, atol=atol))
        self.assertTrue(torch.allclose(pccas.Psi2_diag, pccav.Psi2_diag, atol=atol))
