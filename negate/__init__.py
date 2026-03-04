# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any

__all__ = [
    "Blurb",
    "InferContext",
    "Spec",
    "build_train_call",
    "chart_decompositions",
    "compute_weighted_certainty",
    "configure_runtime_logging",
    "end_processing",
    "infer_origin",
    "load_metadata",
    "load_spec",
    "pretrain",
    "root_folder",
    "run_training_statistics",
    "save_features",
    "save_train_result",
    "train_model",
    "training_loop",
]

_ATTR_SOURCES = {
    "Blurb": ("negate.io.blurb", "Blurb"),
    "InferContext": ("negate.inference", "InferContext"),
    "Spec": ("negate.io.spec", "Spec"),
    "build_train_call": ("negate.train", "build_train_call"),
    "chart_decompositions": ("negate.metrics.track", "chart_decompositions"),
    "compute_weighted_certainty": ("negate.metrics.heuristics", "compute_weighted_certainty"),
    "end_processing": ("negate.io.save", "end_processing"),
    "infer_origin": ("negate.inference", "infer_origin"),
    "load_metadata": ("negate.io.spec", "load_metadata"),
    "load_spec": ("negate.io.spec", "load_spec"),
    "pretrain": ("negate.train", "pretrain"),
    "root_folder": ("negate.io.config", "root_folder"),
    "run_training_statistics": ("negate.metrics.track", "run_training_statistics"),
    "save_features": ("negate.io.save", "save_features"),
    "save_train_result": ("negate.io.save", "save_train_result"),
    "train_model": ("negate.train", "train_model"),
    "training_loop": ("negate.train", "training_loop"),
}

_LOGGING_CONFIGURED = False


def configure_runtime_logging() -> None:
    """Apply quiet logging defaults for third-party ML stacks."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        from datasets import logging as ds_logging, disable_progress_bars as ds_disable_progress_bars
        from diffusers.utils import logging as df_logging
        from huggingface_hub import logging as hf_logging
        from huggingface_hub.utils.tqdm import disable_progress_bars as hf_disable_progress_bars
        from timm.utils.log import setup_default_logging
        from transformers import logging as tf_logging
    except Exception:
        # Keep startup resilient when optional deps are absent.
        _LOGGING_CONFIGURED = True
        return

    setup_default_logging(logging.ERROR)
    for logger in [df_logging, ds_logging, hf_logging, tf_logging]:
        logger.set_verbosity_error()

    ds_disable_progress_bars()
    hf_disable_progress_bars()
    _LOGGING_CONFIGURED = True


def __getattr__(name: str) -> Any:
    source = _ATTR_SOURCES.get(name)
    if source is None:
        raise AttributeError(name)

    module_name, attr_name = source
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
