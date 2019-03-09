"""=============================================================================
Autoencoder with tanh nonlinearities.
============================================================================="""

import numpy as np
from   torch import nn

# ------------------------------------------------------------------------------

class AETanH(nn.Module):

    def __init__(self, cfg):
        super(AETanH, self).__init__()

        assert cfg.GENE_EMBED_DIM < 12
        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE
        self.input_dim = cfg.N_GENES

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, cfg.GENE_EMBED_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.GENE_EMBED_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.input_dim)
        )

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        return self.encoder(x)

# ------------------------------------------------------------------------------

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, self.input_dim)

# ------------------------------------------------------------------------------

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
