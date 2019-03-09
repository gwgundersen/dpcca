"""=============================================================================
Multimodal MNIST data set.
============================================================================="""

import numpy as np
import torch
from   torch.utils.data import Dataset

# ------------------------------------------------------------------------------

class MnistDataset(Dataset):

    def __init__(self, cfg):
        self.config = cfg
        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)

        self.images = data['images'].unsqueeze(1) / 255.
        self.labels = data['labels']
        self.genes  = data['genes']

        assert len(self.images) == self.config.N_SAMPLES
        assert len(self.labels) == self.config.N_SAMPLES
        assert len(self.genes)  == self.config.N_SAMPLES

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in data set.
        """
        return self.config.N_SAMPLES

# ------------------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return the `idx`-th (image, pseudogene)-pair from the dataset.
        """
        image = self.images[idx]
        gene  = self.genes[idx]

        gmin = gene.min()
        gmax = gene.max()
        gene = gene - gmin / (gmax - gmin)

        return image, gene

# ------------------------------------------------------------------------------

    @property
    def n_classes(self):
        """Return number of unique labels.
        """
        return len(np.unique(self.labels))
