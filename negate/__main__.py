# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Negate CLI entry point for training and inference.\n
:returns: None."""

import shutil
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
    chart_decompositions,
    WaveletContext,
    result_path,
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

    print("Prediction selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)

    # show_statistics(features_dataset=features_dataset, start_ns=start_ns)
    # expanded_data = compare_decompositions(model_name=model_name, features_dataset=features_dataset)
    timecode = timer_module.perf_counter() - start_ns
    result_path.mkdir(parents=True, exist_ok=True)
    config_name = "config.toml"
    shutil.copy(str(Path(__file__).parent.parent / "config" / config_name), str(result_path / config_name))

    chart_decompositions(features_dataset=features_dataset, model_name=model_name, vae_name=spec.model_config.auto_vae)
    print(timecode)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""
    spec = Spec()
    dataset_blurb = "Genunie/Human-original dataset path"
    calibrate_blurb = "Check model on the dataset at the provided path from CLI or config, default `assets/`."
    model_blurb = f"Model to use. Default :{spec.model}"  # type: ignore
    model_choices = [repo for repo in spec.models]

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    predict_parser = subparsers.add_parser("predict", help=calibrate_blurb)
    predict_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    predict_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "predict":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None

            multi_prediction(file_or_folder_path=dataset_location, model_name=args.model, spec=spec)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
