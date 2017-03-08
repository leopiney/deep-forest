import logging
import numpy as np
import uuid


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


#
# Taken from: http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter
#
def rolling_window_lastaxis(a, window):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>
    """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)

    return a
