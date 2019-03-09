"""=============================================================================
Utility functions for easy and pretty file logging.
============================================================================="""

import logging
import numpy as np
import types
import torch

# ------------------------------------------------------------------------------

MAIN_LOGGER = 'logger.main'
DIRECTORY   = None

# ------------------------------------------------------------------------------

def set_logfiles(directory, level=logging.INFO):
    """Function setup as many loggers as you want.
    """
    global DIRECTORY
    DIRECTORY = directory

    handler = logging.FileHandler('%s/out.txt' % directory)
    logger = logging.getLogger(MAIN_LOGGER)
    logger.setLevel(level)
    logger.addHandler(handler)

# ------------------------------------------------------------------------------

def log(msg):
    """Print message.
    """
    _log(msg, MAIN_LOGGER)

# ------------------------------------------------------------------------------

def _log(msg, logger_name):
    """Print message to appropriate logger if on cluster, otherwise print to
    stdout.
    """
    if torch.cuda.is_available():
        logger = logging.getLogger(logger_name)
        logger.info(msg)
    else:
        print(msg)

# ------------------------------------------------------------------------------

def log_line(epoch, train_msgs, test_msgs):
    """Print main line in log file, including current epoch and train and test
    data.
    """
    train = '\t'.join(['{:6f}' for _ in train_msgs]).format(*train_msgs)
    test  = '\t'.join(['{:6f}' for _ in test_msgs]).format(*test_msgs)
    msg   = '\t|\t'.join([str(epoch), train, test])
    log(msg)

# ------------------------------------------------------------------------------

def log_section(msg, delim='='):
    """Print message with header.
    """
    log(delim * 80)
    log(msg)
    log(delim * 80)

# ------------------------------------------------------------------------------

def log_args(args):
    """Print arguments passed to script.
    """
    fields = [f for f in vars(args)]
    longest = max(fields, key=len)
    format_str = '{:>%s}  {:}' % len(longest)
    for f in fields:
        msg = format_str.format(f, getattr(args, f))
        log(msg)

# ------------------------------------------------------------------------------

def log_config(cfg):
    """Print settings in the configuration object.
    """
    fields = [f for f in dir(cfg) if not f.startswith('__') and
              type(getattr(cfg, f)) != types.MethodType]
    longest = max(fields, key=len)
    format_str = '{:>%s}  {:}' % len(longest)
    for f in fields:
        if type(getattr(cfg, f)) != types.MethodType:
            msg = format_str.format(f, getattr(cfg, f))
        log(msg)

# ------------------------------------------------------------------------------

def log_model(model):
    """Print model specifications.
    """
    log(model)

# ------------------------------------------------------------------------------

def save_test_indices(indices):
    """Save a Python list so we know our random split of test indices.
    """
    np.save('%s/testset_indices' % DIRECTORY, indices)
