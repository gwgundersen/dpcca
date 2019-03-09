"""============================================================================
Configuration for the multimodal MNIST data set.
============================================================================"""

import matplotlib.pyplot as plt
import torch
from   torchvision.utils import save_image

from   data.config import Config
from   data.mnist.dataset import MnistDataset
from   models import LeNet5AE, AETanH

# ------------------------------------------------------------------------------

class MnistConfig(Config):

    ROOT_DIR       = 'data/mnist'
    N_SAMPLES      = 18623
    N_CHANNELS     = 1
    IMG_SIZE       = 28
    IMG_EMBED_DIM  = 8
    GENE_EMBED_DIM = 8
    N_PIXELS       = 28 * 28
    N_GENES        = 100

# ------------------------------------------------------------------------------

    def get_image_net(self):
        return LeNet5AE(self)

# ------------------------------------------------------------------------------

    def get_genes_net(self):
        return AETanH(self)

# ------------------------------------------------------------------------------

    def get_dataset(self, **kwargs):
        return MnistDataset(self)

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

        # Visualize "genes".
        # ------------------
        x2r = x2r.detach().cpu().numpy()
        fig, ax = plt.subplots()
        ax.scatter(x2r[:, 0], x2r[:, 1], c='blue', marker='.')
        fname = '%s/sample_genes_%s.png' % (directory, desc)
        plt.savefig(fname)
        plt.close('all')
        plt.clf()

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

        x1_fpath = '%s/recon_images_%s.png' % (directory, desc)
        N = min(x.size(0), 8)
        recon = x_recon.view(-1, nc, w, w)[:N]
        x = x.view(-1, nc, w, w)[:N]
        comparison = torch.cat([x, recon])
        save_image(comparison.cpu(), x1_fpath, nrow=N)

# ------------------------------------------------------------------------------

    def save_genes_comparison(self, directory, x, xr, desc):
        x    = x.detach().cpu().numpy()
        xr   = xr.detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1],   c='blue', marker='.')
        ax.scatter(xr[:, 0], xr[:, 1], c='cyan', marker='*')

        fpath = '%s/recon_genes_%s.png' % (directory, str(desc))
        plt.savefig(fpath)
        plt.close('all')
        plt.clf()

# ------------------------------------------------------------------------------

    def save_image_samples(self, directory, model, epoch, x1):
        nc = self.N_CHANNELS
        w  = self.IMG_SIZE

        x1_fpath = '%s/recon_images_%s.png' % (directory, epoch)
        N = min(x1.size(0), 8)
        recon = x1.view(-1, nc, w, w)[:N]
        x = x.view(-1, nc, w, w)[:N]
        comparison = torch.cat([x, recon])
        save_image(comparison.cpu(), x1_fpath, nrow=N)