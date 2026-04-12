# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""CLI logger configuration for the Negate package."""

from __future__ import annotations

import logging
import warnings
from typing import Any

__all__ = ["CLI_LOGGER", "configure_runtime_logging", "get_cli_logger", "set_root_folder"]

ROOT_FOLDER = None  # type: ignore


def get_cli_logger() -> logging.Logger:
    """Get or create the CLI logger with StreamHandler.

    :returns: Configured CLI logger instance.
    """

    logger = logging.getLogger("negate.cli")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


CLI_LOGGER = get_cli_logger()


def set_root_folder(root_folder) -> None:
    """Set the root folder path for logger configuration.

    :param root_folder: Path object representing the root folder.
    """

    global ROOT_FOLDER
    ROOT_FOLDER = root_folder


def configure_runtime_logging() -> None:
    """Apply quiet logging defaults for third-party ML stacks.

    Silences progress bars and sets verbosity to error level for optional dependencies.
    """

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        from datasets import logging as ds_logging, disable_progress_bars as ds_disable_progress_bars
        from diffusers.utils import logging as df_logging
        from huggingface_hub import logging as hf_logging
        from huggingface_hub.utils.tqdm import disable_progress_bars as hf_disable_progress_bars
        from timm.utils.log import setup_default_logging
        from transformers import logging as tf_logging
    except ImportError:
        return

    setup_default_logging(logging.ERROR)
    for logger in [df_logging, ds_logging, hf_logging, tf_logging]:
        logger.set_verbosity_error()

    ds_disable_progress_bars()
    hf_disable_progress_bars()
