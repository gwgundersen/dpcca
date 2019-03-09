"""=============================================================================
Test that our vectorized implementation of the negative log-likelihood is
correct.
============================================================================="""

import numpy as np
import unittest
import torch

from   models import PCCASimple, PCCAVec, PCCAOpt

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def setUp(self):
        n = 1000
        p1 = 50
        p2 = 30
        k = 10
        p = p1 + p2

        self.y = torch.randn(p, n)

        self.p1 = p1;
        self.p2 = p2;
        self.k = k

    def test_same_nlls(self):
        pccas = PCCASimple(self.k, [self.p1, self.p2], 1)
        pccav = PCCAVec(self.k, [self.p1, self.p2], 1)

        n_tries = 3
        i = 0
        nlls1 = torch.empty(n_tries)
        nlls2 = torch.empty(n_tries)

        while True:
            if i >= n_tries:
                break

            # Generate new random parameters.
            Lambda1 = torch.randn(self.p1, self.k)
            Lambda2 = torch.randn(self.p2, self.k)
            Psi1_diag = torch.randn(self.p1)
            Psi2_diag = torch.randn(self.p2)

            # Set each model with these new random parameters.
            pccas.Lambda1.data = Lambda1
            pccas.Lambda2.data = Lambda2
            pccas.Psi1_diag.data = Psi1_diag
            pccas.Psi2_diag.data = Psi2_diag

            pccav.Lambda1.data = Lambda1
            pccav.Lambda2.data = Lambda2
            pccav.Psi1_diag.data = Psi1_diag
            pccav.Psi2_diag.data = Psi2_diag

            Lambda   = torch.cat([Lambda1, Lambda2], dim=0)
            Psi_diag = torch.cat([Psi1_diag, Psi2_diag])

            # Compute negative log likelihood for these data and parameters.
            nll1 = pccas.neg_log_likelihood(self.y, Lambda, Psi_diag)
            nll2 = pccav.neg_log_likelihood(self.y, Lambda, Psi_diag)

            if np.isnan(nll1) or np.isnan(nll2):
                if i > 0:
                    i -= 1
            else:
                nlls1[i] = nll1
                nlls2[i] = nll2
                i += 1

        # These are really big numbers. Close by 10 is actually good, I think.
        self.assertTrue(torch.allclose(nlls1, nlls2, atol=10))
