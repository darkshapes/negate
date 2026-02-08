# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Negate CLI entry point for training and inference.\n
:returns: None."""

import argparse
from pathlib import Path
from sys import argv

from datasets import Dataset

from negate import (
    # grade,
    WaveletAnalyzer,
    build_datasets,
    show_statistics,
    model_config,
    compare_decompositions,
)


def calibration(model_name: str, file_or_folder_path: Path | None = None) -> None:
    """Calibration of computing wavelet energy features.\n
    :param path: Dataset root folder."""

    print("Calibration selected.")
    dataset: Dataset = build_datasets(file_or_folder_path)
    analyzer = WaveletAnalyzer(model_name)
    features_dataset = analyzer.decompose(dataset)
    # grade(result_dataset)
    features_dataset.set_format(type="pandas", columns=["label", "sim_min", "sim_max", "idx_min", "idx_max"])
    show_statistics(features_dataset=features_dataset)
    compare_decompositions(model_name=model_name, features_dataset=features_dataset)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    dataset_blurb = "Genunie/Human-original dataset path"
    train_blurb = "Train XGBoost model on wavelet features using the dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    calibrate_blurb = "Check model on the dataset at the provided path from CLI or config, default `assets/`."
    model_blurb = f"Model to use. Default :{model_config.auto_model[0]}"  # type: ignore
    model_choices = [repo for repo in model_config.list_models]
    check_blurb = "Check whether an image at the provided path is synthetic or original."
    synthetic_blurb = "Mark image as synthetic (label = 1) for evaluation."
    genuine_blurb = "Mark image as genuine (label = 0) for evaluation."

    calibrate_parser = subparsers.add_parser("calibrate", help=calibrate_blurb)
    calibrate_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    calibrate_parser.add_argument("-m", "--model", choices=model_choices, default=model_config.auto_model[0], help=model_blurb)  # type: ignore
    train_parser = subparsers.add_parser("train", help=train_blurb)
    train_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    args = parser.parse_args()

    check_parser = subparsers.add_parser("check", help=check_blurb)
    check_parser.add_argument("path", help="Image or folder path")
    label_grp = check_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help=synthetic_blurb)
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help=genuine_blurb)
    subparsers.add_parser("compare", help="Run extraction and training using all possible VAE.")
    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "calibrate":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None

            calibration(file_or_folder_path=dataset_location, model_name=args.model)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")

            # predict(Path(args.path), true_label=args.label)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
