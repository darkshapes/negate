# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Command-line interface entry point for Negate package.\n
Handles CLI parsing, dataset loading, preprocessing, and result saving.
Supports 'predict' subcommand with automatic timestamping.

Dataset (images)
    → io/ops (load)
    → wavelet.py (decompose)
    → feature_{vit,vae}.py + residuals.py (extract)
    → train.py (XGBoost grade)
    → track.py (plotting/metrics)
"""

import argparse
import json
import os
import re
import time as timer_module
import tomllib
from pathlib import Path
from sys import argv
from typing import Any

import pandas as pd
from datasets import Dataset

from negate import (
    InferContext,
    Spec,
    build_datasets,
    chart_decompositions,
    classify_gnf_or_syn,
    end_processing,
    generate_dataset,
    grade,
    infer_origin,
    load_config_options,
    result_path,
    root_folder,
    run_training_statistics,
    save_features,
    save_train_result,
    wavelet_preprocessing,
)
from negate.io.config import NegateConfig
from negate.train import get_time

start_ns = timer_module.perf_counter()


def build_train_call(args: argparse.Namespace, path_result: Path, spec: Spec) -> Dataset:
    """Prepare CLI command input for function call.\n
    :param args: Parsed command-line arguments.
    :param path_result: Directory containing training outputs.
    :param spec: Model specification container.
    :returns: A dataset of raw images or precalculated features
    """
    kwargs = {}
    if args.features is not None:
        file_feat = str(path_result / args.features / f"features_{args.features}.json")
        with open(file_feat) as handle:
            features = json.load(handle)
        features_df = pd.DataFrame.from_dict(features)
        features_ds: Dataset = Dataset.from_pandas(features_df)
    else:
        kwargs["genuine_path"] = args.path
        if args.syn is not None:
            kwargs["synthetic_path"] = Path(args.syn)
        spec.model = args.model
        try:
            spec.vae = next(iter(x for x in spec.model_config.list_vae if args.ae in x))
        except StopIteration:
            raise ValueError(f"Invalid VAE choice: {args.ae}")
        kwargs["spec"] = spec
        origin_ds: Dataset = build_datasets(**kwargs)
        features_ds = pretrain(origin_ds, spec)
    return features_ds


def load_spec(model_version: Path = Path("config")) -> Spec:
    """Load model specification and training metadata.\n
    :param ver_model: Version folder path containing config and results.
    :returns: Updated specification and additional metadata
    """

    if str(model_version) != "config":
        path_result = Path("results") / model_version.stem
    else:
        path_result = model_version
    path_config = str(path_result / "config.toml")
    config_options = load_config_options(path_config)  # load a different config
    spec = Spec(*config_options)
    return spec


def fetch_spec_data(model_version: Path = Path("config")) -> dict[str, Any]:  # unpack metadata, change individual options
    """Load configuration from TOML file in results or config folder.\n
    :param model_version: Subfolder name under results, defaults to 'config'.
    :returns: Dictionary of loaded configuration values."""

    path_conf = root_folder
    if str(model_version) != "config":
        path_result = str(path_conf / "results" / model_version.stem / "config.toml")
    else:
        path_result = str(path_conf / model_version / "config.toml")
    with open(path_result, "rb") as handle:
        metadata = tomllib.load(handle)
    return metadata


def load_metadata(model_version: Path) -> dict[str, Any]:
    """\nLoad serialized training metadata from JSON result file.\n
    :param model_version: Stem of the model version folder.
    :returns: Dictionary containing saved metrics and parameters."""

    results_path = root_folder / "results" / model_version.stem / f"results_{model_version.stem}.json"
    with open(results_path, "rb") as handle:
        metadata = json.load(handle)
    return metadata


def adjust_spec(metadata: dict[str, Any], hyper_param: str | None = None, param_value: int | float | None = None) -> Spec:
    """Reconstruct spec with optional hyperparameter override.
    :param metadata: Base configuration dictionary.
    :param hyper_param: Key name of parameter to modify.
    :param param_value: New value for the hyperparameter.
    :returns: Reconstructed specification object.
    """
    for label in ["model", "vae", "param", "datasets", "library", "rounds", hyper_param]:
        metadata.pop(label)
    config_replacement = NegateConfig(**{str(hyper_param): param_value}, **metadata)
    config_options = load_config_options()
    spec = Spec(config_replacement, *config_options[1:])

    return spec


def pretrain(image_ds: Dataset, spec: Spec) -> Dataset:
    """Calibration of computing wavelet energy features.\n
    :param ds_orig: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :returns: Dataset of extracted image features
    """
    features_ds = wavelet_preprocessing(image_ds, spec=spec)
    end_processing("Pretraining", start_ns)
    save_features(features_ds)
    chart_decompositions(features_dataset=features_ds, spec=spec)
    return features_ds


def train_model(spec: Spec, features_ds: Dataset) -> None:
    """Train XGBoost model on preprocessed image features.\n
    :param spec: Specification container.
    """
    train_result = grade(features_ds, spec)
    timecode = end_processing("Training", start_ns)
    save_train_result(train_result)
    run_training_statistics(train_result=train_result, timecode=timecode, spec=spec)


def training_loop(image_ds: Dataset, spec: Spec) -> None:
    """Train models across a range of hyperparameter values.\n
    :param ds_orig: Input dataset for training.
    """
    print("looping")

    def parse_num(val):
        """Try int first, fallback to float."""
        try:
            return int(val)
        except ValueError:
            return float(val)

    hyper_param = input("enter name of hyperparameter:")
    step_val = parse_num(input("enter increment"))
    start, end = map(parse_num, input("enter start and end values separated by comma").split(","))

    param_value = start
    metadata = fetch_spec_data()
    spec = adjust_spec(metadata=metadata, param_value=param_value, hyper_param=hyper_param)
    while param_value < end:
        path_loop = root_folder / "results" / get_time()
        features_ds = pretrain(image_ds=image_ds, spec=spec)
        train_model(features_ds=features_ds, spec=spec)
        os.rename(result_path, path_loop)
        param_value += step_val


def main() -> None:
    """CLI argument parser and command dispatcher.\n
    :raises ValueError: Missing image path.
    :raises ValueError: Invalid VAE choice.
    :raises NotImplementedError: Unsupported command passed.
    """
    pretrain_blurb = "Analyze and graph performance of image preprocessing on the image dataset at the provided path from CLI and config paths, default `assets/`."
    train_blurb = "Train XGBoost model on preprocessed image features using the image dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    infer_blurb = "Infer whether an image at the provided path is synthetic or original."
    dataset_blurb = "Genunie/Human-original image dataset path"
    synthetic_blurb = "Synthetic image dataset path"
    unidentified_blurb = "Path to the image or directory containing images of unidentified origin"
    calculate_blurb = "Measure defining features of synthetic or genuine images at the provided path."
    spec = Spec()

    model_blurb = f"Model to use. Default :{spec.model}"
    model_choices = [repo for repo in spec.models]
    ae_choices = [ae[0] for ae in spec.model_config.list_vae]
    ae_choices.append("")
    models_path = root_folder / "models"
    results_path = root_folder / "results"

    list_results = []
    if len(os.listdir(results_path)) > 0:
        list_results = [str(folder.stem) for folder in Path(results_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_results.sort(reverse=True)

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help=pretrain_blurb)
    train_parser = subparsers.add_parser("train", help=train_blurb)
    train_parser.add_argument("-l", "--loop", action="store_true", help="Loop training iterations across hyperparameter settings")
    train_parser.add_argument("-f", "--features", choices=list_results, default=None, help="Train from an existing set of features")

    calculate_parser = subparsers.add_parser("calculate", help=calculate_blurb)
    calculate_parser.add_argument("path", help=unidentified_blurb)

    infer_parser = subparsers.add_parser("infer", help=infer_blurb)
    infer_parser.add_argument("path", help=unidentified_blurb)
    if len(os.listdir(models_path)) > 0:
        list_model = [str(folder.stem) for folder in Path(models_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_model.sort(reverse=True)
        if list_model:
            infer_parser.add_argument("-m", "--model", choices=list_model, default=list_model[0])
    else:
        list_model = None
        infer_parser.add_argument("-m", "--model", choices=None, default=None)

    for sub in [pretrain_parser, train_parser]:
        sub.add_argument("path", help=dataset_blurb, nargs="?", default=None)
        sub.add_argument("-s", "--syn", help=synthetic_blurb, nargs="?", default=None)
        sub.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)
        sub.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)

    for sub in [infer_parser, calculate_parser]:
        label_grp = sub.add_mutually_exclusive_group()
        label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help="Mark image as synthetic (label = 1) for evaluation.")
        label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help="Mark image as genuine (label = 0) for evaluation.")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "pretrain":
            origin_ds = build_train_call(args=args, path_result=results_path, spec=spec)
            pretrain(origin_ds, spec)
        case "train":
            origin_ds = build_train_call(args=args, path_result=results_path, spec=spec)
            if args.loop is True:
                training_loop(image_ds=origin_ds, spec=spec)
            else:
                train_model(features_ds=origin_ds, spec=spec)

        case "infer":
            if args.path is None:
                raise ValueError("Infer requires an image path.")
            if list_model is None or not list_model:
                raise ValueError(f"Warning: No valid model directories found in {models_path} Create or add a trained model before running inference.")
            file_image: Path = Path(args.path)
            model_version: Path = models_path / args.model
            if not model_version.exists():
                raise ValueError("Model format must match pattern YYYYMMDD_HHMMSS (e.g., 20240101_123456)")
            print(f"""Checking path '{file_image}' using model date {model_version.stem}""")

            origin_ds: Dataset = generate_dataset(file_image)

            infer_context = InferContext(
                spec=load_spec(model_version),
                model_version=model_version,
                train_metadata=load_metadata(model_version),
                label=args.label,
            )
            infer_origin(infer_context, wavelet_preprocessing(origin_ds, spec))
        case "calculate":
            if args.path is None:
                raise ValueError("Calculating the origin requires an image path.")

            file_image: Path = Path(args.path)
            origin_ds: Dataset = generate_dataset(file_image)
            features_ds = wavelet_preprocessing(origin_ds, spec=spec)
            json_path = save_features(features_ds)
            chart_decompositions(features_dataset=features_ds, spec=spec)
            probabilities = classify_gnf_or_syn(json_path)
            print(origin_ds.description)
            print(probabilities)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
