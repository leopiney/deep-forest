import logging
import uuid


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger
