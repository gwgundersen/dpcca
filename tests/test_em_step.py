"""=============================================================================
Test vectorized EM vs. simple EM.
============================================================================="""

import torch
import unittest

from   models import PCCASimple, PCCAVec, PCCAOpt

# ------------------------------------------------------------------------------

inv   = torch.inverse
outer = torch.ger

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def setUp(self):
        self.n = 1000
        p1 = 5
        p2 = 3

        self.k = 2
        self.p = p1 + p2

        # Initialize parameters to verify that both implementations compute the
        # same values.
        self.P_diag = torch.diag(torch.randn(self.p, self.p)) * 10
        self.L      = torch.randn(self.p, self.k) * 10
        self.y      = torch.randn(self.p, self.n) * 10

        # We only initialize these in order to access their methods.
        self.pccas = PCCASimple(latent_dim=self.k, dims=[p1, p2], n_iters=1)
        self.pccav = PCCAVec(latent_dim=self.k, dims=[p1, p2], n_iters=1)
        self.pccao = PCCAOpt(latent_dim=self.k, dims=[p1, p2], n_iters=1,
                             private_z=False)

    def test_em_step(self):
        atol = 0.1
        L_new1, P_diag_new1 = self.pccas.em_step(self.y, self.L, self.P_diag)
        L_new2, P_diag_new2 = self.pccav.em_step(self.y, self.L, self.P_diag)
        L_new3, P_diag_new3 = self.pccav.em_step(self.y, self.L, self.P_diag)
        self.assertTrue(torch.allclose(L_new1, L_new2, atol=atol))
        self.assertTrue(torch.allclose(L_new2, L_new3, atol=atol))
        self.assertTrue(torch.allclose(P_diag_new1, P_diag_new2, atol=atol))
        self.assertTrue(torch.allclose(P_diag_new2, P_diag_new3, atol=atol))
