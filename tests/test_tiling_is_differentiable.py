"""=============================================================================
Verify that my mental model of how PyTorch handles nn.Parameters is correct.
============================================================================="""

import unittest

from   copy import deepcopy
import torch
from   torch import nn, optim
from   torch.nn import functional as F

import cuda

# ------------------------------------------------------------------------------

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        device = cuda.device()

        self.params1 = nn.Parameter(torch.randn(3, 5, device=device))
        self.params2 = nn.Parameter(torch.randn(3, 5, device=device))

    def forward(self, x):
        params = torch.cat([self.params1, self.params2])
        assert type(params) is torch.Tensor
        y = torch.matmul(params, x)
        return y

# ------------------------------------------------------------------------------

class UnitTest(unittest.TestCase):

    def test_parameters_simple(self):
        device = cuda.device()
        params = nn.Parameter(torch.randn(3, 5, device=device))
        optimizer = optim.Adam([params], lr=0.1)
        params_saved = deepcopy(params.data)
        x = torch.ones(5, device=device)
        y = params @ x
        t = torch.zeros(3, device=device)
        loss = F.mse_loss(t, y)
        loss.backward()
        optimizer.step()

        # We expect the parameters to change.
        self.assertFalse(bool((params.data == params_saved).all()))

    def test_tiled_params(self):
        device = cuda.device()
        model = Model()

        optimizer = optim.Adam(model.parameters(), lr=0.1)
        params1_saved = deepcopy(model.params1.data)
        params2_saved = deepcopy(model.params2.data)

        x = torch.ones(5, device=device)
        y = model.forward(x)
        t = torch.zeros(6, device=device)
        loss = F.mse_loss(t, y)
        loss.backward()
        optimizer.step()

        # # We expect the parameters to change.
        self.assertFalse(bool((model.params1.data == params1_saved).all()))
        self.assertFalse(bool((model.params2.data == params2_saved).all()))

    def test_untiled_params(self):
        pass
