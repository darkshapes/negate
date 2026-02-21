# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Command-line interface entry point for the Negate package.

This module provides the CLI parser and main execution logic for training
and inference workflows. It handles argument parsing, dataset loading,
preprocessing orchestration, and result saving.

The CLI supports a 'predict' subcommand for running inference on image datasets.
Results are automatically timestamped and saved to the results directory.

Functions:
    preprocessing: Apply wavelet analysis transformations to dataset.
    multi_prediction: Main prediction workflow orchestrator.
    main: CLI argument parser and command dispatcher.
"""

from itertools import accumulate
import shutil
import argparse
import datetime
import time as timer_module
from pathlib import Path
from sys import argv

from datasets import Dataset
from torch import accelerator

from negate import (
    # generate_dataset,
    # load_remote_dataset,
    Spec,
    WaveletAnalyze,
    build_datasets,
    chart_decompositions,
    WaveletContext,
    result_path,
    grade,
)
from negate.track import accuracy


start_ns = timer_module.perf_counter()
process_time = lambda: print(str(datetime.timedelta(seconds=timer_module.process_time())), end="")


def preprocessing(dataset: Dataset, spec: Spec) -> Dataset:
    """Apply wavelet decomposition analysis to dataset images.\n
    :param dataset: HuggingFace Dataset containing 'image' column with PIL images.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column containing wavelet coefficients.\n
    :raises KeyError: If dataset missing required 'image' column.
    :raises RuntimeError: If wavelet analyzer fails to initialize.

    .. note::
        The preprocessing is memory-efficient and supports batched processing
        when config.toml `batch_size` > 0.
    """
    kwargs = {}
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size

    context = WaveletContext(spec)
    with WaveletAnalyze(context) as analyzer:  # type: ignore
        dataset = dataset.map(
            analyzer,
            remove_columns=["image"],
            desc="Computing wavelets...",
            **kwargs,
        )
    result_path.mkdir(parents=True, exist_ok=True)
    config_name = "config.toml"
    shutil.copy(str(Path(__file__).parent.parent / "config" / config_name), str(result_path / config_name))

    return dataset


def metric_analysis(spec: Spec, file_or_folder_path: Path | None = None) -> None:
    """Calibration of computing wavelet energy features.\n
    :param path: Dataset root folder."""

    print("Metrics selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)
    timecode = timer_module.perf_counter() - start_ns
    chart_decompositions(features_dataset=features_dataset)
    print(f"Metrics completed in {timecode}")
    print(timecode)


def train_model(spec: Spec, file_or_folder_path: Path | None = None) -> None:
    print("Training selected.")

    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)
    train_result = grade(features_dataset, spec)
    timecode = timer_module.perf_counter() - start_ns
    print(f"Training completed in {timecode}")
    accuracy(train_result=train_result, timecode=timecode)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""

    metrics_blurb = "Analyze performance of image preprocessing on the dataset at the provided path from CLI and config paths, default `assets/`."
    train_blurb = "Train XGBoost model on wavelet features using the dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    dataset_blurb = "Genunie/Human-original dataset path"

    spec = Spec()

    model_blurb = f"Model to use. Default :{spec.model}"  # type: ignore
    model_choices = [repo for repo in spec.models]
    ae_choices = [ae[0] for ae in spec.model_config.list_vae]
    ae_choices.append("")

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    metrics_parser = subparsers.add_parser("metrics", help=metrics_blurb)
    metrics_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    metrics_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
    metrics_parser.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)  # type: ignore

    train_parser = subparsers.add_parser("train", help=train_blurb)
    train_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    train_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
    train_parser.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)  # type: ignore

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "metrics":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None
            model_name = (args.model,)
            model_name = (args.model,)
            metric_analysis(file_or_folder_path=dataset_location, spec=spec)
        case "train":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None
            metric_analysis(file_or_folder_path=dataset_location, spec=spec)

        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
