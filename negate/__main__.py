# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Negate CLI entry point for training and inference.\n
:returns: None."""

import argparse
import datetime
import time as timer_module
from pathlib import Path
from sys import argv

from datasets import Dataset

from negate import (
    # generate_dataset,
    # load_remote_dataset,
    Spec,
    WaveletAnalyze,
    build_datasets,
    compare_decompositions,
    WaveletContext,
)


start_ns = timer_module.perf_counter()
process_time = lambda: print(str(datetime.timedelta(seconds=timer_module.process_time())), end="")


def preprocessing(dataset: Dataset, spec: Spec) -> Dataset:
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
    return dataset


def multi_prediction(model_name: str, spec: Spec, file_or_folder_path: Path | None = None) -> None:
    """Calibration of computing wavelet energy features.\n
    :param path: Dataset root folder."""

    print("Calibration selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    features_dataset = preprocessing(dataset, spec=spec)

    # show_statistics(features_dataset=features_dataset, start_ns=start_ns)
    print(features_dataset.info)
    compare_decompositions(model_name=model_name, features_dataset=features_dataset)
    timecode = timer_module.perf_counter() - start_ns


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""
    spec = Spec()
    dataset_blurb = "Genunie/Human-original dataset path"
    calibrate_blurb = "Check model on the dataset at the provided path from CLI or config, default `assets/`."
    model_blurb = f"Model to use. Default :{spec.model}"  # type: ignore
    model_choices = [repo for repo in spec.models]
    # check_blurb = "Check whether an image at the provided path is synthetic or original."
    # synthetic_blurb = "Mark image as synthetic (label = 1) for evaluation."
    # genuine_blurb = "Mark image as genuine (label = 0) for evaluation."
    # mixed_blurb = "Mark images as mixed for evaluation."
    # train_blurb = "Train XGBoost model on wavelet features using the dataset in the provided path or `assets/`. The resulting model will be saved to disk."

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # subparsers.add_parser("compare", help="Run extraction and training using all possible VAE.")

    # train_parser = subparsers.add_parser("train", help=train_blurb)
    # train_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)

    predict_parser = subparsers.add_parser("predict", help=calibrate_blurb)
    predict_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    predict_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore

    # check_parser = subparsers.add_parser("check", help=check_blurb)
    # check_parser.add_argument("path", help="Image or folder path", nargs="?", default=None)
    # check_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
    # label_grp = check_parser.add_mutually_exclusive_group()
    # label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help=synthetic_blurb)
    # label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help=genuine_blurb)
    # label_grp.add_argument("-x", "--mixed", action="store_const", const=0, dest="label", help=mixed_blurb)

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "predict":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None

            multi_prediction(file_or_folder_path=dataset_location, model_name=args.model, spec=spec)
        # case "check":
        #     if args.label is None:
        #         if args.path is None:
        #             raise ValueError("Check requires an image path.")
        #         start_check(
        #             model_name=args.model,
        #             file_or_folder_path=Path(args.path),
        #         )
        #     else:
        #         kwargs = {}
        #         if args.path is not None:
        #             kwargs.setdefault("file_or_folder_path", Path(args.path))
        #         start_evaluation(
        #             model_name=args.model,
        #             true_label=args.label,
        #             **kwargs,
        #         )
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
