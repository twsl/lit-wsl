import logging

from lightning.pytorch.utilities import rank_zero_only

LOG_LEVELS = (
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "fatal",
)


def get_logger(name: str) -> logging.Logger:
    """Initializes standard python command line logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(ch)
    return logger


def get_rank_zero_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = get_logger(name)

    for level in LOG_LEVELS:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
