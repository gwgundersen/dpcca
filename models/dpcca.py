"""=============================================================================
Deep probabilistic CCA (DPCCA) for histology images and gene expression levels.
============================================================================="""

import torch
from   torch import nn

from   models import PCCA
import cuda

# ------------------------------------------------------------------------------

class DPCCA(nn.Module):

    def __init__(self, cfg, latent_dim, em_iters=1):
        """Initialize Deep Probabilistic CCA model.
        """
        super(DPCCA, self).__init__()

        if latent_dim >= cfg.IMG_EMBED_DIM or latent_dim >= cfg.N_GENES:
            msg = 'The latent dimension must be smaller than both the image '\
                  'embedding dimensions and genes dimension.'
            raise AttributeError(msg)

        self.cfg = cfg
        self.image_net  = cfg.get_image_net()
        self.genes_net  = cfg.get_genes_net()
        self.latent_dim = latent_dim

        self.pcca = PCCA(latent_dim=latent_dim,
                         dims=[cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                         max_iters=em_iters)

        # This initialization is pulled from the DCGAN implementation:
        #
        #    https://github.com/pytorch/examples/blob/master/dcgan/main.py
        #
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass of images and associated signal through model.
        """
        y = self.encode(x)
        y1r, y2r = self.pcca.forward(y)
        x1r = self.image_net.decode(y1r)
        x2r = self.genes_net.decode(y2r)
        return x1r, x2r

# ------------------------------------------------------------------------------

    def encode(self, x):
        """Embed data in preparation for PCCA.
        """
        x1, x2 = x

        y1 = self.image_net.encode(x1)
        y2 = self.genes_net.encode(x2)

        # PCCA assumes our data is mean-centered.
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)

        y = torch.cat([y1, y2], dim=1)

        # PCCA expects (p-dims, n-samps)-dimensional data.
        y = y.t()

        return y

# ------------------------------------------------------------------------------

    def sample(self, x, n_samples=None):
        """Sample from fitted PCCA-VAE model.
        """
        x1, x2 = x
        y1 = self.image_net.encode(x1)
        y2 = self.genes_net.encode(x2)

        if not n_samples:
            n_samples = x1.shape[0]

        return self._sample(y1, y2, n_samples, False)

# ------------------------------------------------------------------------------

    def sample_x1_from_x2(self, x2):
        """Sample images based on gene expression data.
        """
        device = cuda.device()
        y1 = torch.zeros(x2.shape[0], self.cfg.IMG_EMBED_DIM, device=device)
        y2 = self.genes_net.encode(x2)
        x1r, _ = self._sample(y1, y2, n_samples=None, sample_across=True)
        return x1r

# ------------------------------------------------------------------------------

    def sample_x2_from_x1(self, x1):
        """Sample gene expression data from images.
        """
        device = cuda.device()
        y1 = self.image_net.encode(x1)
        y2 = torch.zeros(x1.shape[0], self.cfg.GENE_EMBED_DIM, device=device)
        _, x2r = self._sample(y1, y2, n_samples=None, sample_across=True)
        return x2r

# ------------------------------------------------------------------------------

    def _sample(self, y1, y2, n_samples, sample_across):
        """Utility function for all sampling methods. Takes a pair of embeddings
        and returns a pair of reconstructed samples.
        """
        assert not y1.requires_grad
        assert not y2.requires_grad
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)
        y  = torch.cat([y1, y2], dim=1)
        y  = y.t()
        if sample_across:
            y1r, y2r = self.pcca.sample(y, one_sample_per_y=True)
        else:
            y1r, y2r = self.pcca.sample(y, n_samples=n_samples)
        x1r = self.image_net.decode(y1r)
        x2r = self.genes_net.decode(y2r)

        return x1r, x2r

# ------------------------------------------------------------------------------

    def estimate_z_given_x(self, x):
        """Estimate the latent variable z given our data x.
        """
        y = self.encode(x)
        return self.pcca.estimate_z_given_y(y).t()

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, x):
        """Compute the negative log-likelihood of the data given our current
        parameters.
        """
        y = self.encode(x)
        return self.pcca.neg_log_likelihood(y)
