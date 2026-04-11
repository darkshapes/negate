# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""CLI logger configuration for the Negate package."""

from __future__ import annotations

import logging

ROOT_FOLDER = None  # type: ignore


def get_cli_logger() -> logging.Logger:
    """Get or create the CLI logger with StreamHandler.\n
    :returns: Configured CLI logger instance."""

    logger = logging.getLogger("negate.cli")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_root_folder(root_folder) -> None:
    """Set the root folder path for logger configuration.\n
    :param root_folder: Path object representing the root folder."""

    global ROOT_FOLDER
    ROOT_FOLDER = root_folder
