"""=============================================================================
Linear autoencoder.
============================================================================="""

from   torch import nn

# ------------------------------------------------------------------------------

class AELinear(nn.Module):

    def __init__(self, cfg):
        """Initialize simple linear model.
        """
        super(AELinear, self).__init__()
        self.input_dim = cfg.N_GENES
        emb_dim  = cfg.GENE_EMBED_DIM
        self.fc1 = nn.Linear(self.input_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, self.input_dim)

# ------------------------------------------------------------------------------

    def encode(self, x):
        return self.fc1(x)

# ------------------------------------------------------------------------------

    def decode(self, z):
        return self.fc2(z)

# ------------------------------------------------------------------------------

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)
