"""
utils/logger.py – Centralised logging configuration for the pipeline.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a consistent format for the given *name*.

    If a handler is already attached to the root logger (e.g. from
    ``logging.basicConfig``), the returned logger inherits that configuration.
    Otherwise a new ``StreamHandler`` writing to *stdout* is attached.
    """
    logger = logging.getLogger(name)
    if not logging.root.handlers and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
