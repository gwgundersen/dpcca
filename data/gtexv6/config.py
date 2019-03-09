"""============================================================================
Configuration for the GTEx V6 data set.
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models import DCGANAE128, AELinear
from   data.gtexv6.dataset import GTExV6Dataset
from   data.config import Config

# ------------------------------------------------------------------------------

class GTExV6Config(Config):

    ROOT_DIR       = 'data/gtexv6'
    N_SAMPLES      = 2221
    IMG_SIZE       = 128
    N_CHANNELS     = 3
    N_PIXELS       = 3 * 128 * 128
    N_GENES        = 18659
    IMG_EMBED_DIM  = 1000
    GENE_EMBED_DIM = 1000

# ------------------------------------------------------------------------------

    def get_image_net(self):
        return DCGANAE128(self)

# ------------------------------------------------------------------------------

    def get_genes_net(self):
        return AELinear(self)

# ------------------------------------------------------------------------------

    def get_dataset(self, **kwargs):
        return GTExV6Dataset(self)

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc, x1, x2, labels):
        n_samples = 64
        nc = self.N_CHANNELS
        w = self.IMG_SIZE

        # Visualize images.
        # -----------------
        x1r, x2r = model.sample([x1, x2], n_samples)
        x1r = x1r.view(n_samples, nc, w, w)
        fname = '%s/sample_images_%s.png' % (directory, desc)
        save_image(x1r.cpu(), fname)

# ------------------------------------------------------------------------------

    def save_image_samples(self, directory, model, desc, x1):
        n_samples = 64
        nc = self.N_CHANNELS
        w = self.IMG_SIZE

        x1_recon, _ = model.sample(None, n_samples)
        x1_recon = x1_recon.view(n_samples, nc, w, w)
        fname = '%s/sample_%s.png' % (directory, desc)
        save_image(x1_recon.cpu(), fname)

# ------------------------------------------------------------------------------

    def save_comparison(self, directory, x, x_recon, desc, is_x1=None):
        """Save image samples from learned image likelihood.
        """
        if is_x1:
            self.save_image_comparison(directory, x, x_recon, desc)
        else:
            self.save_genes_comparison(directory, x, x_recon, desc)

# ------------------------------------------------------------------------------

    def save_image_comparison(self, directory, x, x_recon, desc):
        nc = self.N_CHANNELS
        w  = self.IMG_SIZE

        x1_fpath = '%s/%s_images_recon.png' % (directory, desc)
        N = min(x.size(0), 8)
        recon = x_recon.view(-1, nc, w, w)[:N]
        x = x.view(-1, nc, w, w)[:N]
        comparison = torch.cat([x, recon])
        save_image(comparison.cpu(), x1_fpath, nrow=N)

# ------------------------------------------------------------------------------

    def save_genes_comparison(self, directory, x, xr, desc):
        n, _ = x.shape
        x    = x.detach().cpu().numpy()
        xr   = xr.detach().cpu().numpy()

        x_cov  = np.cov(x)
        xr_cov = np.cov(xr)

        comparison = np.hstack([x_cov, xr_cov])
        plt.imshow(comparison)

        fpath = '%s/%s_genes_recon.png' % (directory, str(desc))
        plt.savefig(fpath)
        plt.close('all')
        plt.clf()
