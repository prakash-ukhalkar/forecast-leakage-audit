"""Standard logging configuration for this project.

All modules should use `get_logger(__name__)` instead of `print`.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured stdlib logger.

    The log level is read from the `SDG_LEAKAGE_AUDIT_LOG_LEVEL` environment
    variable (e.g. "DEBUG", "INFO", "WARNING"), defaulting to "INFO" if unset or
    invalid. A single console (stream) handler is attached exactly once per
    logger name to avoid duplicate log lines on repeated calls (e.g. from
    notebook cell re-execution).

    Args:
        name: Logger name, conventionally `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        level_name = os.environ.get("SDG_LEAKAGE_AUDIT_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False

        _CONFIGURED_LOGGERS.add(name)

    return logger
