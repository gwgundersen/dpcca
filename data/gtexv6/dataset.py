"""=============================================================================
GTEx V6 data set of histology images and gene expression levels.
============================================================================="""

from   sklearn import preprocessing
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms

# ------------------------------------------------------------------------------

class GTExV6Dataset(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg

        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)
        self.images     = data['images']
        self.genes      = data['genes']
        self.names      = data['fnames']
        self.tissues    = data['tissues']
        self.gene_names = data['gnames']

        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)

        # `classes` are the unique class names, i.e. tissues.
        self.classes = list(set(self.tissues))

        self.subsample_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((0, 360)),
            transforms.RandomCrop(cfg.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        return len(self.images)

# ------------------------------------------------------------------------------

    def __getitem__(self, i):
        """Return the `idx`-th (image, metadata)-pair from the dataset.
        """
        pixels = self.images[i]
        genes  = self.genes[i]

        bad_crop = True
        while bad_crop:
            image = self.subsample_image(pixels).numpy()
            # We want to avoid all black crops because it prevents us from
            # feature normalization.
            if image.min() == image.max():
                continue
            # We want to avoid crops that are majority black.
            if (image == 0).sum() / image.size > 0.5:
                continue

            bad_crop = False
            image = torch.Tensor(image)

        return image, genes

# ------------------------------------------------------------------------------

    def sample_ith_image(self, i):
        """Memory-saving utility function when we just want an image.
        """
        return self.subsample_image(self.images[i])

# ------------------------------------------------------------------------------

    def get_all_tissue(self, label, test_inds):
        """Return all samples of a specific tissue.
        """
        if type(label) is str:
            label = int(self.labelEncoder.transform([label])[0])

        n = 0
        for i in test_inds:
            if self.labels[i] == label:
                n += 1

        nc = self.cfg.N_CHANNELS
        w  = self.cfg.IMG_SIZE
        images = torch.Tensor(n, nc, w, w)

        for i in test_inds:
            if self.labels[i] == label:
                images[i] = self.subsample_image(self.images[i])

        if type(label) is str:
            label = int(self.labelEncoder.transform([label])[0])

        inds = torch.Tensor(self.labels) == label
        inds = inds and test_inds
        images_raw = self.images[inds]

        n  = len(images_raw)
        nc = self.cfg.N_CHANNELS
        w  = self.cfg.IMG_SIZE

        images = torch.Tensor(n, nc, w, w)
        for i, img in enumerate(images_raw):
            images[i] = self.subsample_image(img)

        genes  = self.genes[inds]
        return images, genes
