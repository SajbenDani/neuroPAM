"""
logging_utils.py -- Shared logging configuration for the harmonization package.

Every module obtains its logger through :func:`get_logger`; the CLI calls
:func:`configure_logging` once so that console output and an optional
run-log file receive the same records.  Messages are ASCII-only so they
render on Windows consoles with legacy code pages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_NAME = "vae_harmonization"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under the package root."""
    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{_ROOT_NAME}.{short}")


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    stream=None,
) -> logging.Logger:
    """
    Attach a console handler (and optionally a file handler) to the package
    root logger.  Safe to call more than once: existing handlers are replaced.
    """
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream or sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.propagate = False
    return root
