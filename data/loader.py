"""=============================================================================
Dataset-agnostic data loader.
============================================================================="""

import math
import numpy as np
import random

from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data import DataLoader

from   data import GTExV6Config
from   data import MnistConfig

# ------------------------------------------------------------------------------

def get_config(dataset):
    """Return configuration object based on dataset string.
    """
    SUPPORTED_DATASETS = ['gtexv6', 'mnist']

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset %s is not supported.' % dataset)
    if dataset == 'gtexv6':
        return GTExV6Config()
    if dataset == 'mnist':
        return MnistConfig()

# ------------------------------------------------------------------------------

def get_data_loaders(cfg, batch_size, num_workers, pin_memory, cv_pct=None,
                     directory=None):
    """Return dataset and return data loaders for train and CV sets.
    """
    if cv_pct is not None and directory is not None:
        msg = 'Both CV % and a directory cannot both be specified.'
        raise ValueError(msg)
    if cv_pct is not None and cv_pct >= 1.0:
        raise ValueError('`CV_PCT` should be strictly less than 1.')

    dataset = cfg.get_dataset()
    indices = list(range(len(dataset)))

    if directory:
        test_inds  = list(np.load('%s/testset_indices.npy' % directory))
        train_inds = list(set(indices) - set(test_inds))
    else:
        random.shuffle(indices)  # Shuffles in-place.
        split      = math.floor(len(dataset) * (1 - cv_pct))
        train_inds = indices[:split]
        test_inds  = indices[split:]

    # If batch_size == -1, then we want full batches.
    train_batch_size = batch_size if batch_size != -1 else len(train_inds)
    test_batch_size  = batch_size if batch_size != -1 else len(test_inds)
    assert train_batch_size == test_batch_size

    # If data set size is indivisible by batch size, drop last incomplete batch.
    # Dropping the last batch is fine because we randomly subsample from the
    # data set, meaning all data should be sampled uniformly in expectation.
    DROP_LAST = True

    # This is Soumith's recommended approach. See:
    #
    #     https://github.com/pytorch/pytorch/issues/1106
    #
    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_inds),
        batch_size=train_batch_size,
        num_workers=num_workers,
        drop_last=DROP_LAST,

        # Move loaded and processed tensors into CUDA pinned memory. See:
        #
        #     http://pytorch.org/docs/master/notes/cuda.html
        #
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(test_inds),
        batch_size=test_batch_size,
        num_workers=num_workers,
        drop_last=DROP_LAST,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
