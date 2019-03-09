"""=============================================================================
Test vectorized Psi_diag_new vs. simple Psi_diag_new.
============================================================================="""

import torch
import unittest

from   models import PCCAVec, PCCASimple
import linalg as LA

# ------------------------------------------------------------------------------

diag  = torch.diag
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

# ------------------------------------------------------------------------------

    def test_Psi_rterm(self):
        Psi_rterm1 = torch.zeros(self.p, self.p)
        for yi in self.y.t():
            Ez1 = self.pccas.E_z_given_y(self.L, self.P_diag, yi)
            Psi_rterm1 += outer(yi, yi) - outer(self.L @ Ez1, yi)

        Ez2 = self.pccav.E_z_given_y(self.L, self.P_diag, self.y)
        Psi_rterm2 = LA.sum_outers(self.y, self.y) - LA.sum_outers(self.L @ Ez2,
                                                                   self.y)

        self.assertTrue(torch.allclose(Psi_rterm1, Psi_rterm2, atol=0.01))

# ------------------------------------------------------------------------------

    # In principle we should test this, but we're literally just diagonalizing
    # and multiplying by 1./n. It has nothing to do with vectorization.
    def test_Psi_diag_new(self):
        Psi_rterm1 = torch.zeros(self.p, self.p)
        for yi in self.y.t():
            Ez1 = self.pccas.E_z_given_y(self.L, self.P_diag, yi)
            Psi_rterm1 += outer(yi, yi) - outer(self.L @ Ez1, yi)
        Psi_new1 = 1./self.n * diag(Psi_rterm1)

        Ez2 = self.pccav.E_z_given_y(self.L, self.P_diag, self.y)
        Psi_rterm2 = LA.sum_outers(self.y, self.y) - LA.sum_outers(self.L @ Ez2,
                                                                   self.y)
        Psi_new2 = 1./self.n * diag(Psi_rterm2)

        self.assertTrue(torch.allclose(Psi_new1, Psi_new2, atol=0.01))
