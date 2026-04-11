# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import importlib
from typing import Any
from negate.io.console import configure_runtime_logging
from negate.io.console import get_cli_logger
from negate.io.console import set_root_folder

__all__ = [
    "Blurb",
    "InferContext",
    "Spec",
    "build_train_call",
    "chart_decompositions",
    "compute_weighted_certainty",
    "end_processing",
    "get_cli_logger",
    "infer_origin",
    "load_metadata",
    "load_spec",
    "pretrain",
    "root_folder",
    "run_training_statistics",
    "save_features",
    "save_train_result",
    "set_root_folder",
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
    "get_cli_logger": ("negate.io.logging", "get_cli_logger"),
    "infer_origin": ("negate.inference", "infer_origin"),
    "load_metadata": ("negate.io.spec", "load_metadata"),
    "load_spec": ("negate.io.spec", "load_spec"),
    "pretrain": ("negate.train", "pretrain"),
    "root_folder": ("negate.io.config", "root_folder"),
    "run_training_statistics": ("negate.metrics.track", "run_training_statistics"),
    "save_features": ("negate.io.save", "save_features"),
    "save_train_result": ("negate.io.save", "save_train_result"),
    "set_root_folder": ("negate.io.logging", "set_root_folder"),
    "train_model": ("negate.train", "train_model"),
    "training_loop": ("negate.train", "training_loop"),
}


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
