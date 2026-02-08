# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Negate CLI entry point for training and inference.\n
:returns: None."""

import argparse
from pathlib import Path
from sys import argv

import numpy as np
from datasets import Dataset

from negate import (
    WaveletAnalyzer,
    build_datasets,
    compare_decompositions,
)


def calibration(file_or_folder_path: Path | None = None, compare: bool = False) -> None:
    """Calibration of computing wavelet energy features.\n
    :param path: Dataset root folder."""

    print("Calibration selected.")
    dataset: Dataset = build_datasets(file_or_folder_path)
    analyzer = WaveletAnalyzer()
    result_dataset = analyzer.decompose(dataset)
    result_dataset.set_format(type="pandas", columns=["label", "sim_min", "sim_max", "idx_min", "idx_max"])

    genuine_dataset = result_dataset.filter(lambda x: x["label"] == 0, batched=False)
    synthetic_dataset = result_dataset.filter(lambda x: x["label"] == 1, batched=False)

    avg_genuine_sim = float(np.mean(genuine_dataset["sim_min"]))
    avg_synthetic_sim = float(np.mean(synthetic_dataset["sim_min"]))

    print(f"Average similarity min (genuine): {avg_genuine_sim:.4f}")
    print(f"Average similarity min (synthetic): {avg_synthetic_sim:.4f}")

    avg_genuine_sim = float(np.mean(genuine_dataset["sim_max"]))
    avg_synthetic_sim = float(np.mean(synthetic_dataset["sim_max"]))

    print(f"Average similarity max (genuine): {avg_genuine_sim:.4f}")
    print(f"Average similarity max (synthetic): {avg_synthetic_sim:.4f}")

    avg_genuine_sim = float(np.mean(genuine_dataset["idx_min"]))
    avg_synthetic_sim = float(np.mean(synthetic_dataset["idx_min"]))

    print(f"Average perturbed min (genuine): {avg_genuine_sim:.4f}")
    print(f"Average perturbed min (synthetic): {avg_synthetic_sim:.4f}")

    avg_genuine_sim = float(np.mean(genuine_dataset["idx_max"]))
    avg_synthetic_sim = float(np.mean(synthetic_dataset["idx_max"]))

    print(f"Average perturbed max (genuine): {avg_genuine_sim:.4f}")
    print(f"Average perturbed max (synthetic): {avg_synthetic_sim:.4f}")

    idx_overall_avg = np.mean((genuine_dataset["idx_min"])) + np.mean(genuine_dataset["idx_max"]) / 2
    print(f"Overall average genuine cosine similarity: {idx_overall_avg:.4f}")
    sim_overall_average = np.mean((genuine_dataset["sim_min"])) + np.mean(genuine_dataset["sim_max"]) / 2
    print(f"Overall average genuine cosine similarity: {sim_overall_average:.4f}")
    g_avg = (idx_overall_avg + sim_overall_average) / 2
    overall_avg = np.mean((synthetic_dataset["idx_min"])) + np.mean(synthetic_dataset["idx_max"]) / 2
    print(f"Overall average synthetic cosine similarity: {overall_avg:.4f}")
    sim_overall_average = np.mean((synthetic_dataset["sim_min"])) + np.mean(synthetic_dataset["sim_max"]) / 2
    print(f"Overall average synthetic cosine similarity: {sim_overall_average:.4f}")
    s_avg = (idx_overall_avg + sim_overall_average) / 2
    print(f"Overall average cosine similarity: {(s_avg + g_avg / 2):.4f}")
    compare_decompositions(result_dataset)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="Check model on the dataset at the provided path from CLI or config, default `assets/`.")
    calibrate_parser.add_argument("path", help="Genunie/Human-original dataset path", nargs="?", default=None)

    check_parser = subparsers.add_parser(
        "check",
        help="Check whether an image at the provided path is synthetic or original.",
    )
    check_parser.add_argument("path", help="Image or folder path")
    label_grp = check_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help="Mark image as synthetic (label = 1) for evaluation.")
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help="Mark image as genuine (label = 0) for evaluation.")
    subparsers.add_parser("compare", help="Run extraction and training using all possible VAE.")
    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "calibrate":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None
            calibration(file_or_folder_path=dataset_location)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")

            # predict(Path(args.path), true_label=args.label)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
