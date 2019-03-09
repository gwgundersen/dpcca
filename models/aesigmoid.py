"""=============================================================================
Autoencoder with sigmoid nonlinearities.
============================================================================="""

import torch
from   torch import nn
from   torch.nn import functional as F

# ------------------------------------------------------------------------------

class AESigmoid(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(AESigmoid, self).__init__()

        assert cfg.EMBEDDING_DIM < 12
        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE
        self.input_dim = self.nc * self.w * self.w

        mid_dim = cfg.VAE_MID_DIM
        if 'pcca_z_dim' in kwargs:
            latent_dim = kwargs['pcca_z_dim']
        else:
            latent_dim = cfg.EMBEDDING_DIM

        self.fc1 = nn.Linear(self.input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim,        latent_dim)

        self.fc3 = nn.Linear(latent_dim,     mid_dim)
        self.fc4 = nn.Linear(mid_dim,        self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = x.view(-1, self.input_dim)
        h = F.relu(self.fc1(x))
        return self.fc2(h)

# ------------------------------------------------------------------------------

    def decode(self, z):
        h = F.relu(self.fc3(z))
        xr = torch.sigmoid(self.fc4(h))
        xr = xr.view(-1, self.nc, self.w, self.w)
        return xr

# ------------------------------------------------------------------------------

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
