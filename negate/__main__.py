# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Command-line interface entry point for Negate package.\n
Handles CLI parsing, dataset loading, preprocessing, and result saving.
Supports 'inference' and 'train' subcommand with automatic timestamping.

    → Dataset (images)
    → io/ops (load)
    → wavelet.py (decompose)
    → feature_{vit,vae}.py + residuals.py (extract)
    → train.py (XGBoost grade)
    → inference.py (predict from data)
    → track.py (plotting/metrics)
"""

import argparse
import os
import re
import time as timer_module
from pathlib import Path
from sys import argv

from negate import (
    InferContext,
    Spec,
    build_train_call,
    chart_decompositions,
    compute_weighted_certainty,
    end_processing,
    infer_origin,
    load_metadata,
    load_spec,
    pretrain,
    root_folder,
    run_training_statistics,
    save_features,
    save_train_result,
    train_model,
    training_loop,
)

start_ns = timer_module.perf_counter()


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

    for sub in [pretrain_parser, train_parser]:
        sub.add_argument("path", help=dataset_blurb, nargs="?", default=None)
        sub.add_argument("-s", "--syn", help=synthetic_blurb, nargs="?", default=None)
        sub.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)
        sub.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)

    infer_parser = subparsers.add_parser("infer", help=infer_blurb)
    infer_parser.add_argument("path", help=unidentified_blurb)
    if len(os.listdir(models_path)) > 0:
        list_model = [str(folder.stem) for folder in Path(models_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_model.sort(reverse=True)
        if list_model:
            infer_parser.add_argument(
                "-m",
                "--model",
                choices=list_model,
                default=[
                    "20260225_185933",  # FLUX-AE
                    "20260225_221149",  # DC-AE
                ],
            )
    else:
        list_model = None
        infer_parser.add_argument("-m", "--model", choices=None, default=None)
    infer_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose console output")

    label_grp = infer_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help="Mark image as synthetic (label = 1) for evaluation.")
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help="Mark image as genuine (label = 0) for evaluation.")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "pretrain":
            origin_ds = build_train_call(args=args, path_result=results_path, spec=spec)
            features_ds = pretrain(origin_ds, spec)
            end_processing("Pretraining", start_ns)
            save_features(features_ds)
            chart_decompositions(features_dataset=features_ds, spec=spec)
        case "train":
            origin_ds = build_train_call(args=args, path_result=results_path, spec=spec)
            if args.loop is True:
                training_loop(image_ds=origin_ds, spec=spec)

            else:
                train_result = train_model(features_ds=origin_ds, spec=spec)
                timecode = end_processing("Training", start_ns)
                save_train_result(train_result)
                run_training_statistics(train_result=train_result, timecode=timecode, spec=spec)

        case "infer":
            model_pair_blurb = "Two models must be provided for inference (e.g., [20240101_123456, 20251231_987654]"
            model_pattern_blurb = "Model format must match pattern YYYYMMDD_HHMMSS (e.g., 20240101_123456)"
            if args.path is None:
                raise ValueError("Infer requires an image path.")
            if list_model is None or not list_model:
                raise ValueError(f"Warning: No valid model directories found in {models_path} Create or add a trained model before running inference.")
            img_file_or_folder: Path = Path(args.path)
            assert isinstance(args.model, list) or isinstance(args.model, tuple), ValueError(model_pair_blurb)
            ae_model: Path = models_path / args.model[0]
            dc_model: Path = models_path / args.model[1]
            assert ae_model and ae_model.exists(), ValueError(model_pattern_blurb)
            assert dc_model and dc_model.exists(), ValueError(model_pattern_blurb)
            if args.verbose:
                print(f"""Checking path '{img_file_or_folder}' using models dated {args.model}""")

            context_ae = InferContext(
                spec=load_spec(ae_model),
                model_version=ae_model,
                train_metadata=load_metadata(ae_model),
                label=args.label,
                file_or_folder_path=img_file_or_folder,
                dataset_feat=None,
                dc_vae=False,
                verbose=args.verbose,
            )
            context_dc = InferContext(
                spec=load_spec(dc_model),
                model_version=dc_model,
                train_metadata=load_metadata(dc_model),
                label=args.label,
                file_or_folder_path=img_file_or_folder,
                dataset_feat=None,
                dc_vae=True,
                verbose=args.verbose,
            )

            ae_inference = infer_origin(context_ae)
            dc_inference = infer_origin(context_dc)

            inferences = compute_weighted_certainty(
                ae_inference,
                dc_inference,
                args.label,
            )

        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
