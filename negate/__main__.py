# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Negate CLI entry point for training and inference.\n
:returns: None."""

import argparse
import datetime
import time as timer_module
from pathlib import Path
from sys import argv
import numpy as np

from datasets import Dataset

from negate import (
    # grade,
    WaveletAnalyzer,
    build_datasets,
    compare_decompositions,
    generate_dataset,
    load_remote_dataset,
    model_config,
    negate_d,
    show_statistics,
)

start_ns = timer_module.perf_counter()
process_time = lambda: print(str(datetime.timedelta(seconds=timer_module.process_time())), end="")


def start_check(
    model_name: str,
    file_or_folder_path: Path,
):
    dataset: Dataset = generate_dataset(folder_path=file_or_folder_path)
    analyzer = WaveletAnalyzer(model_name)
    features_dataset = analyzer.decompose(dataset)
    features_dataset.set_format(type="pandas", columns=["label", "sim_min", "sim_max", "idx_min", "idx_max"])
    show_statistics(features_dataset=features_dataset, start_ns=start_ns)
    compare_decompositions(model_name=model_name, features_dataset=features_dataset)
    timecode = timer_module.perf_counter() - start_ns
    print(timecode)


def start_evaluation(
    model_name: str,
    true_label: int | None,
    file_or_folder_path: Path | None = None,
) -> None:
    """Process images of the dataset measuring computed wavelet energy features.\n
    :param path: Dataset root folder."""
    print(f"""{"Evaluation" if true_label is not None else "Detection"} selected.
Checking path '{file_or_folder_path}' with {model_name}""")

    if file_or_folder_path is not None:
        dataset: Dataset = generate_dataset(folder_path=file_or_folder_path, label=true_label)
    else:
        dataset_path = Path(__file__).parent.parent / ".datasets"
        eval_data = negate_d.eval_data[0]
        dataset: Dataset = load_remote_dataset(repo=eval_data, split="test", folder_path=dataset_path)

    analyzer = WaveletAnalyzer(model_name)
    features_dataset = analyzer.decompose(dataset)
    features_dataset.set_format(type="pandas", columns=["sim_min", "sim_max", "idx_min", "idx_max"])
    data_frame = features_dataset.to_pandas()
    data_frame["sensitivity"] = (data_frame["sim_min"].values + data_frame["sim_max"].values) / 2  # type:ignore
    lower_bound = 0.800
    upper_bound = 0.900
    data_frame["is_within_range"] = (data_frame["sensitivity"] >= lower_bound) & (data_frame["sensitivity"] <= upper_bound)  # type:ignore
    for index, row in data_frame.iterrows():  # type:ignore
        sensitivity_value = float(row["sensitivity"].item()) if isinstance(row["sensitivity"], np.ndarray) else float(row["sensitivity"])  # type: ignore
        status = "Within range : Synthetic" if row["is_within_range"] else "Outside range"  # type:ignore
        print(f"Image {index}: Sensitivity={sensitivity_value:.3f}, Status: {status}")

    synthetic_count = int(data_frame["is_within_range"].sum())  # type:ignore
    non_synthetic_count = len(data_frame) - synthetic_count  # type:ignore
    ratio = synthetic_count / non_synthetic_count if non_synthetic_count > 0 else float("inf")
    print(f"Synthetic to non-synthetic ratio: {ratio:.3f}")

    print(f"{process_time()}")


def calibrate(
    model_name: str,
    file_or_folder_path: Path | None = None,
) -> None:
    """Calibration of computing wavelet energy features.\n
    :param path: Dataset root folder."""

    print("Calibration selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path)
    analyzer = WaveletAnalyzer(model_name)
    features_dataset = analyzer.decompose(dataset)
    features_dataset.set_format(type="pandas", columns=["label", "sim_min", "sim_max", "idx_min", "idx_max"])
    show_statistics(features_dataset=features_dataset, start_ns=start_ns)
    compare_decompositions(model_name=model_name, features_dataset=features_dataset)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    dataset_blurb = "Genunie/Human-original dataset path"
    # train_blurb = "Train XGBoost model on wavelet features using the dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    calibrate_blurb = "Check model on the dataset at the provided path from CLI or config, default `assets/`."
    model_blurb = f"Model to use. Default :{model_config.auto_model[0]}"  # type: ignore
    model_choices = [repo for repo in model_config.list_models]
    check_blurb = "Check whether an image at the provided path is synthetic or original."
    synthetic_blurb = "Mark image as synthetic (label = 1) for evaluation."
    genuine_blurb = "Mark image as genuine (label = 0) for evaluation."
    mixed_blurb = "Mark images as mixed for evaluation."

    calibrate_parser = subparsers.add_parser("calibrate", help=calibrate_blurb)
    calibrate_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    calibrate_parser.add_argument("-m", "--model", choices=model_choices, default=model_config.auto_model[0], help=model_blurb)  # type: ignore
    # train_parser = subparsers.add_parser("train", help=train_blurb)
    # train_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)

    check_parser = subparsers.add_parser("check", help=check_blurb)
    check_parser.add_argument("path", help="Image or folder path", nargs="?", default=None)
    check_parser.add_argument("-m", "--model", choices=model_choices, default=model_config.auto_model[0], help=model_blurb)  # type: ignore
    label_grp = check_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help=synthetic_blurb)
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help=genuine_blurb)
    label_grp.add_argument("-x", "--mixed", action="store_const", const=0, dest="label", help=mixed_blurb)

    # subparsers.add_parser("compare", help="Run extraction and training using all possible VAE.")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "calibrate":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None

            calibrate(file_or_folder_path=dataset_location, model_name=args.model)
        case "check":
            if args.label is None:
                if args.path is None:
                    raise ValueError("Check requires an image path.")
                start_check(
                    model_name=args.model,
                    file_or_folder_path=Path(args.path),
                )
            else:
                kwargs = {}
                if args.path is not None:
                    kwargs.setdefault("file_or_folder_path", Path(args.path))
                start_evaluation(
                    model_name=args.model,
                    true_label=args.label,
                    **kwargs,
                )
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
