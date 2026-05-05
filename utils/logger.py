import logging
import sys


def setup_logger(name="acds-detr"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger

