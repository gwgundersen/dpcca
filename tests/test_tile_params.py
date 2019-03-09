"""=============================================================================
Verify that parameter unrolling function is correct.
============================================================================="""

import unittest

import torch
from   torch import nn

import cuda
from   models import PCCAOpt

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def setUp(self):
        self.k  = 2
        self.p1 = 2
        self.p2 = 3
        self.pcca = PCCAOpt(latent_dim=self.k,
                            dims=[self.p1, self.p2],
                            n_iters=1,
                            private_z=True)

    def test_untile_params(self):
        device = cuda.device()

        L1_before  = torch.ones(self.p1, self.k, device=device)
        L2_before  = torch.ones(self.p2, self.k, device=device) * 2
        B1_before  = torch.ones(self.p1, self.k, device=device) * 3
        B2_before  = torch.ones(self.p2, self.k, device=device) * 4
        B12_before = torch.zeros(self.p1, self.k, device=device)
        B21_before = torch.zeros(self.p2, self.k, device=device)
        P1_before  = torch.ones(self.p1, device=device) * 5
        P2_before  = torch.ones(self.p2, device=device) * 6

        L = torch.cat([
            torch.cat([L1_before, B1_before, B12_before], dim=1),
            torch.cat([L2_before, B21_before, B2_before], dim=1)
        ], dim=0)
        P = torch.cat([P1_before, P2_before])

        L1_after, L2_after, B1_after, B2_after, P1_after, P2_after = \
            self.pcca.untile_params(L, P)

        self.assertTrue((L1_after == L1_before).all())
        self.assertTrue((L2_after == L2_before).all())
        self.assertTrue((B1_after == B1_before).all())
        self.assertTrue((B2_after == B2_before).all())
        self.assertTrue((P1_after == P1_before).all())
        self.assertTrue((P2_after == P2_before).all())

    def test_untile_params_grad(self):
        device = cuda.device()

        # These are the parameters we want to be updated even if we manipulate
        # them after tiling and then unrolling.
        L1_before  = nn.Parameter(torch.ones(self.p1, self.k, device=device))
        L2_before  = nn.Parameter(torch.ones(self.p2, self.k, device=device) * 2)
        B1_before  = nn.Parameter(torch.ones(self.p1, self.k, device=device) * 3)
        B2_before  = nn.Parameter(torch.ones(self.p2, self.k, device=device) * 4)
        B12_before = nn.Parameter(torch.zeros(self.p1, self.k, device=device))
        B21_before = nn.Parameter(torch.zeros(self.p2, self.k, device=device))
        P1_before  = nn.Parameter(torch.ones(self.p1, device=device) * 5)
        P2_before  = nn.Parameter(torch.ones(self.p2, device=device) * 6)

        L = torch.cat([
            torch.cat([L1_before, B1_before, B12_before], dim=1),
            torch.cat([L2_before, B21_before, B2_before], dim=1)
        ], dim=0)
        P = torch.cat([P1_before, P2_before])

        L1_after, L2_after, B1_after, B2_after, P1_after, P2_after = \
            self.pcca.untile_params(L, P)

        # ----------------------------------------------------------------------
        # Gradient before is None because we have never called backward().
        self.assertTrue(L1_before.grad is None)
        x = torch.ones(self.p1)
        x.requires_grad_(True)
        y = L1_after.t() @ x
        y.backward(torch.ones(self.k))
        # Gradient after is not None because manipulating L1_after effects
        # L1_before.
        self.assertTrue(L1_before.grad is not None)
        self.assertFalse((L1_before.grad == 0).all())

        # Gradient before is not None because we have manipulated L2_before via
        # L1_before via L.
        self.assertTrue((L2_before.grad == 0).all())
        x = torch.ones(self.p2)
        x.requires_grad_(True)
        y = L2_after.t() @ x
        y.backward(torch.ones(self.k))
        # The gradient before is not None but it should not be all 0s.
        self.assertTrue(L2_before.grad is not None)
        self.assertFalse((L2_before.grad == 0).all())

        # ----------------------------------------------------------------------
        # Gradient before is not None because we have manipulated B1_before via
        # L1_before and L2_before via L.
        self.assertTrue((B1_before.grad == 0).all())
        x = torch.ones(self.p1)
        x.requires_grad_(True)
        y = B1_after.t() @ x
        y.backward(torch.ones(self.k))
        # The gradient before is not None but it should not be all 0s.
        self.assertTrue(B1_before.grad is not None)
        self.assertFalse((B1_before.grad == 0).all())

        # Gradient before is not None because we have manipulated B2_before via
        # L1_before and L2_before via L.
        self.assertTrue((B2_before.grad == 0).all())
        x = torch.ones(self.p2)
        x.requires_grad_(True)
        y = B2_after.t() @ x
        y.backward(torch.ones(self.k))
        # The gradient before is not None but it should not be all 0s.
        self.assertTrue(B2_before.grad is not None)
        self.assertFalse((B2_before.grad == 0).all())

        # ----------------------------------------------------------------------
        # Gradient before is None because we have not manipulated Psi in any way
        # when manipulating Lambdas and Bs.
        self.assertTrue(P1_before.grad is None)
        x = torch.ones(self.p1)
        x.requires_grad_(True)
        y = P1_after @ x
        y.backward()
        self.assertTrue(P1_before.grad is not None)
        self.assertFalse((P1_before.grad == 0).all())

        self.assertTrue((P2_before.grad == 0).all())
        x = torch.ones(self.p2)
        x.requires_grad_(True)
        y = P2_after @ x
        y.backward()
        self.assertTrue(P2_before.grad is not None)
        self.assertFalse((P2_before.grad == 0).all())
