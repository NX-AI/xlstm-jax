#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Logging setup for the command line interface.

Ported from https://github.com/NX-AI/xlstm-dev/blob/main/xlstm/ml_utils/log_utils/log_cmd.py
"""

import logging
import os
import sys

import jax

LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s"

LOGGER = logging.getLogger(__name__)


def setup_exception_logging():
    """Make sure that uncaught exceptions are logged with the logger."""

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception("Uncaught exception", exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging


def get_loglevel() -> str:
    """Get loglevel from `LOGLEVEL` environment variable.

    Returns:
        str: loglevel
    """
    return os.environ.get("LOGLEVEL", "INFO").upper()


def setup_logging(logfile: str = "output.log"):
    """Initialize logging to `logfile` and `stdout`.

    Args:
        logfile: Name of the log file. Defaults to "output.log".
    """
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # retrieve loglevel from external environment variable, default is INFO
    LOGLEVEL = get_loglevel()
    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        level=LOGLEVEL,
        format=LOG_FORMAT.format(rank=""),
        force=True,
    )

    setup_exception_logging()

    LOGGER.info(f"Logging to {logfile} initialized.")


def setup_logging_multiprocess(logfile: str = "output.log"):
    """Initialize logging to `logfile` and `stdout` for JAX distributed training.

    Args:
        logfile: Name of the log file. Defaults to "output.log".
    """

    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)

    def _setup_subprocess_logger(rank: int, level: str | int) -> None:
        logger = logging.getLogger()
        logger.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT.format(rank=f"[R{rank:d}]"))
        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    # retrieve loglevel from external environment variable, default is INFO
    LOGLEVEL = get_loglevel()
    # give all processes the same handlers
    logging.basicConfig(
        handlers=[stdout_handler, file_handler],
        force=True,
    )
    # set log format (adding the rank) and loglevel depending on rank
    if jax.process_index() == 0:
        # standard logger should log everything
        _setup_subprocess_logger(jax.process_index(), LOGLEVEL)
    else:
        # other processes should only log CRITICAL and higher
        _setup_subprocess_logger(jax.process_index(), logging.ERROR)

    # Setup exception logging for multiprocess (taken from xlstm-dev). We may need to adjust this in future
    # as sys except hook does not work with multiprocessing (see [1]) (Remedy see [2])
    # [1] https://stackoverflow.com/questions/47815850/python-sys-excepthook-on-multiprocess
    # [2] https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    setup_exception_logging()

    LOGGER.info(f"Logging to {logfile} initialized.")
