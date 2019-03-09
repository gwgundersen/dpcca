"""=============================================================================
Test vectorized Lambda_new vs. simple Lambda_new.
============================================================================="""

import torch
import unittest

from   models import PCCASimple, PCCAVec
import linalg as LA

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

# ------------------------------------------------------------------------------

    def test_E_z_given_y(self):
        Ez1 = torch.empty(self.k, self.n)
        for i in range(self.n):
            yi = self.y[:, i]
            Ez1[:, i] = self.pccas.E_z_given_y(self.L, self.P_diag, yi)
        Ez2 = self.pccav.E_z_given_y(self.L, self.P_diag, self.y)
        self.assertTrue(torch.allclose(Ez1, Ez2, atol=0.0001))

# ------------------------------------------------------------------------------

    def test_E_zzT_given_y(self):
        Ezz1 = torch.zeros(self.k, self.k)
        for i in range(self.n):
            yi = self.y[:, i]
            Ezz1 += self.pccas.E_zzT_given_y(self.L, self.P_diag, yi, self.k)
        Ezz2 = self.pccav.E_zzT_given_y(self.L, self.P_diag, self.y, self.k)
        self.assertTrue(torch.allclose(Ezz1, Ezz2, atol=0.0001))

# ------------------------------------------------------------------------------

    def test_Lambda_lterm(self):
        Lambda_lterm1 = torch.zeros(self.p, self.k)
        for yi in self.y.t():
            Ez1 = self.pccas.E_z_given_y(self.L, self.P_diag, yi)
            Lambda_lterm1 += outer(yi, Ez1)

        Ez2 = self.pccav.E_z_given_y(self.L, self.P_diag, self.y)
        Lambda_lterm2 = LA.sum_outers(self.y, Ez2)

        self.assertTrue(torch.allclose(Lambda_lterm1, Lambda_lterm2,
                                       atol=0.0001))

# ------------------------------------------------------------------------------

    def test_Lambda_new(self):
        Lambda_lterm = torch.zeros(self.p, self.k)
        Lambda_rterm = torch.zeros(self.k, self.k)
        for yi in self.y.t():
            Ez = self.pccas.E_z_given_y(self.L, self.P_diag, yi)
            Lambda_lterm += outer(yi, Ez)
            Lambda_rterm += self.pccas.E_zzT_given_y(self.L, self.P_diag, yi,
                                                     self.k)
        Lambda_new1 = Lambda_lterm @ inv(Lambda_rterm)

        Ez2 = self.pccav.E_z_given_y(self.L, self.P_diag, self.y)
        Lambda_lterm2 = torch.einsum('ab,cb->ac', [self.y, Ez2])
        Lambda_rterm2 = self.pccav.E_zzT_given_y(self.L, self.P_diag, self.y,
                                                 self.k)
        Lambda_new2 = Lambda_lterm2 @ inv(Lambda_rterm2)

        # Increasing tolerance because of compounding round off errors.
        self.assertTrue(torch.allclose(Lambda_new1, Lambda_new2, atol=0.001))
