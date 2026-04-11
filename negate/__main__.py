# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""CLI entry point for the Negate package."""

from __future__ import annotations

import argparse
import logging
import re
import time as timer_module
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from sys import argv
from typing import Any

ROOT_FOLDER = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_FOLDER / "config"
BLURB_PATH = CONFIG_PATH / "blurb.toml"
CONFIG_TOML_PATH = CONFIG_PATH / "config.toml"
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")
DEFAULT_INFERENCE_PAIR = ["20260225_185933", "20260225_221149"]
start_ns = timer_module.perf_counter()
CLI_LOGGER = logging.getLogger("negate.cli")
if not CLI_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    CLI_LOGGER.addHandler(_handler)
CLI_LOGGER.setLevel(logging.INFO)
CLI_LOGGER.propagate = False


@dataclass
class BlurbText:
    """CLI help text defaults loaded from config/blurb.toml."""

    pretrain: str = "Analyze and graph performance..."
    train: str = "Train XGBoost model..."
    infer: str = "Infer whether features..."

    loop: str = "Toggle training across the range..."
    features_load: str = "Train from an existing set of features"
    verbose: str = "Verbose console output"
    label_syn: str = "Mark image as synthetic (label = 1) for evaluation."
    label_gne: str = "Mark image as genuine (label = 0) for evaluation."

    gne_path: str = "Genunie/Human-origin image dataset path"
    syn_path: str = "Synthetic image dataset path"
    unidentified_path: str = "Path to the image or directory containing images of unidentified origin"

    verbose_status: str = "Checking path "
    verbose_dated: str = " using models dated "

    infer_path_error: str = "Infer requires an image path."
    model_error: str = "Warning: No valid model directories found in "
    model_error_hint: str = " Create or add a trained model before running inference."
    model_pair: str = "Two models must be provided for inference..."
    model_pattern: str = "Model format must match pattern YYYYMMDD_HHMMSS..."

    model_desc: str = "model to use. Default : "


@dataclass
class ModelChoices:
    """Model and VAE choices inferred from config/config.toml."""

    default_vit: str = ""
    default_vae: str = ""
    model_choices: list[str] = field(default_factory=list)
    ae_choices: list[str] = field(default_factory=list)


@dataclass
class CmdContext:
    """Container for parsed arguments and runtime dependencies."""

    args: argparse.Namespace
    blurb: Any
    spec: Any
    results_path: Path
    models_path: Path
    list_model: list[str] | None


def load_spec(model_version: str | Path = "config") -> Any:
    """Backwards-compatible export used by tests and callers."""

    from negate.io.spec import load_spec as _load_spec

    return _load_spec(str(model_version))


def _list_timestamp_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries = [entry.name for entry in path.iterdir() if entry.is_dir() and TIMESTAMP_PATTERN.fullmatch(entry.name)]
    entries.sort(reverse=True)
    return entries


def _load_blurb_text() -> BlurbText:
    blurb = BlurbText()
    if not BLURB_PATH.exists():
        return blurb

    with open(BLURB_PATH, "rb") as blurb_file:
        data = tomllib.load(blurb_file)

    for key, value in data.items():
        if hasattr(blurb, key):
            setattr(blurb, key, value)
    return blurb


def _load_model_choices() -> ModelChoices:
    choices = ModelChoices()
    if not CONFIG_TOML_PATH.exists():
        return choices

    with open(CONFIG_TOML_PATH, "rb") as config_file:
        data = tomllib.load(config_file)

    model_library = data.get("model", {}).get("library", {})
    for configured_models in model_library.values():
        if isinstance(configured_models, list):
            choices.model_choices.extend(str(model_name) for model_name in configured_models)
        elif isinstance(configured_models, str):
            choices.model_choices.append(configured_models)
    if choices.model_choices:
        choices.default_vit = choices.model_choices[0]

    vae_library = data.get("vae", {}).get("library", {})
    for configured_vae in vae_library.values():
        if isinstance(configured_vae, list) and configured_vae:
            choices.ae_choices.append(str(configured_vae[0]))

    if "" not in choices.ae_choices:
        choices.ae_choices.append("")
    if choices.ae_choices:
        choices.default_vae = choices.ae_choices[0]
    return choices


def _build_parser(blurb: BlurbText, choices: ModelChoices, list_results: list[str], list_model: list[str], inference_pair: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help=blurb.pretrain)
    train_parser = subparsers.add_parser("train", help=blurb.train)
    train_parser.add_argument("-l", "--loop", action="store_true", help=blurb.loop)
    train_parser.add_argument("-f", "--features", choices=list_results, default=None, help=blurb.features_load)

    process_parser = subparsers.add_parser("process", help="Run all decompose/extract module combinations")
    process_parser.add_argument("path", help=blurb.unidentified_path)
    process_parser.add_argument("-v", "--verbose", action="store_true", help=blurb.verbose)
    process_parser.add_argument("--transposed", default=None, help="Comma-separated transposed indices")
    process_parser.add_argument("--combination", default=None, help="Comma-separated module names")
    process_parser.add_argument("--train", choices=["convnext", "xgboost"], default=None, help="Train model after processing")

    vit_help = f"Vison {blurb.model_desc} {choices.default_vit}".strip()
    ae_help = f"Autoencoder {blurb.model_desc} {choices.default_vae}".strip()
    infer_model_help = f"Trained {blurb.model_desc} {inference_pair}".strip()

    for sub in [pretrain_parser, train_parser]:
        sub.add_argument("path", help=blurb.gne_path, nargs="?", default=None)
        sub.add_argument("-s", "--syn", help=blurb.syn_path, nargs="?", default=None)
        if choices.model_choices:
            sub.add_argument("-m", "--model", choices=choices.model_choices, default=choices.default_vit, help=vit_help)
        else:
            sub.add_argument("-m", "--model", default=choices.default_vit, help=vit_help)
        if choices.ae_choices:
            sub.add_argument("-a", "--ae", choices=choices.ae_choices, default=choices.default_vae, help=ae_help)
        else:
            sub.add_argument("-a", "--ae", default=choices.default_vae, help=ae_help)

    infer_parser = subparsers.add_parser("infer", help=blurb.infer)
    infer_parser.add_argument("path", help=blurb.unidentified_path)
    if list_model:
        infer_parser.add_argument("-m", "--model", choices=list_model, default=inference_pair, nargs="+", help=infer_model_help)
    else:
        infer_parser.add_argument("-m", "--model", choices=None, default=None, nargs="+")

    label_grp = infer_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help=blurb.label_gne)
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help=blurb.label_syn)
    infer_parser.add_argument("-v", "--verbose", action="store_true", help=blurb.verbose)

    return parser


you


def main() -> None:
    blurb_text = _load_blurb_text()
    model_choices = _load_model_choices()

    models_path = ROOT_FOLDER / "models"
    results_path = ROOT_FOLDER / "results"
    list_results = _list_timestamp_dirs(results_path)
    list_model = _list_timestamp_dirs(models_path)
    inference_pair = list_model[:2] if len(list_model) >= 2 else DEFAULT_INFERENCE_PAIR

    parser = _build_parser(
        blurb=blurb_text,
        choices=model_choices,
        list_results=list_results,
        list_model=list_model,
        inference_pair=inference_pair,
    )
    args = parser.parse_args(argv[1:])

    from negate.io.blurb import Blurb
    from negate.io.spec import Spec

    spec = Spec()
    blurb = Blurb(spec)

    cmd_context = CmdContext(
        args=args,
        blurb=blurb,
        spec=spec,
        results_path=results_path,
        models_path=models_path,
        list_model=list_model if list_model else None,
    )
    cmd(cmd_context)


if __name__ == "__main__":
    main()
