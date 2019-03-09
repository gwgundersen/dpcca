"""============================================================================
Configuration for dataset.
============================================================================"""

class Config(object):

    def get_image_net(self):
        """Return neural network used for learning from images.
        """
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def get_genes_net(self, linear):
        """Return neural network used for learning from genes.
        """
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def get_dataset(self):
        """Return dataset instance.
        """
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc, x1, x2, labels):
        """Save samples from learned likelihood.
        """
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def save_comparison(self, directory, x, x_recon, desc, is_x1=None):
        """Save comparison of data and reconstructed data.
        """
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def visualize_dataset(self, directory):
        """Optionally visualize dataset. This is useful in some cases to verify
        the data looks like what we expect.
        """
        pass
