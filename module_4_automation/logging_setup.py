from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LOGGER_NAME = "dq_pipeline"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


@contextmanager
def pipeline_logging(level: str, log_path: Optional[Path]) -> Iterator[logging.Logger]:
    """Send pipeline logs to stderr and, optionally, a file; handlers are removed on exit."""
    logger = logging.getLogger(LOGGER_NAME)
    previous_level = logger.level
    formatter = logging.Formatter(LOG_FORMAT)

    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
