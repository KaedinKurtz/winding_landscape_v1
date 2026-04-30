"""Logging configuration.

Console gets INFO+; file gets DEBUG+. Both include timestamps and module names.
Call :func:`configure_logging` once at startup (typically in ``cli.main``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_file: Path | str, verbose: bool = False) -> None:
    """Set up root logger handlers.

    Parameters
    ----------
    log_file : Path
        Path to the run-log file. Will be opened in write mode (overwriting any
        previous run log -- one log per CLI invocation).
    verbose : bool
        If True, console gets DEBUG+ instead of INFO+.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Wipe any handlers from previous configurations (e.g. in tests).
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-40s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(Path(log_file), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the canonical project namespace prefix."""
    if not name.startswith("winding_landscape"):
        name = f"winding_landscape.{name}"
    return logging.getLogger(name)
