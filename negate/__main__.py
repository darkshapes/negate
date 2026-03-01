# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# type: ignore

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

from __future__ import annotations

import argparse
import os
import re
import time as timer_module
from pathlib import Path
from sys import argv
from dataclasses import dataclass
from tqdm import tqdm


from negate import (
    Blurb,
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


@dataclass
class CmdContext:
    """Container for main() arguments passed to cmd()."""

    args: argparse.Namespace
    blurb: Blurb
    spec: Spec
    results_path: Path
    models_path: Path
    list_model: list[str] | None


def cmd(ctx: CmdContext) -> None:  # -> list[dict[str, str | float | int]]
    """Process command arguments\n
    :raises ValueError: Missing image path.
    :raises ValueError: Invalid VAE choice.
    :raises NotImplementedError: Unsupported command passed.
    """
    args = ctx.args
    match args.cmd:
        case "pretrain":
            origin_ds = build_train_call(args=args, path_result=ctx.results_path, spec=ctx.spec)
            features_ds = pretrain(origin_ds, ctx.spec)
            end_processing("Pretraining", start_ns)
            save_features(features_ds)
            chart_decompositions(features_dataset=features_ds, spec=ctx.spec)
        case "train":
            origin_ds = build_train_call(args=args, path_result=ctx.results_path, spec=ctx.spec)
            if args.loop is True:
                training_loop(image_ds=origin_ds, spec=ctx.spec)

            else:
                train_result = train_model(features_ds=origin_ds, spec=ctx.spec)
                timecode = end_processing("Training", start_ns)
                save_train_result(train_result)
                run_training_statistics(train_result=train_result, timecode=timecode, spec=ctx.spec)

        case "infer":
            if args.path is None:
                raise ValueError(ctx.blurb.infer_path_error)
            if ctx.list_model is None or not ctx.list_model:
                raise ValueError(f"{ctx.blurb.model_error} {ctx.models_path} {ctx.blurb.model_error_hint}")
            img_file_or_folder: Path = Path(args.path)
            assert isinstance(args.model, list) or isinstance(args.model, tuple), ValueError(ctx.blurb.model_pair)
            negate_models = {}
            for saved_model in args.model:
                negate_models[saved_model] = ctx.models_path / saved_model
                assert negate_models[saved_model].exists(), ValueError(ctx.blurb.model_pattern)
            if args.verbose:
                import warnings

                warnings.filterwarnings("default", category=UserWarning)
                warnings.filterwarnings("default", category=DeprecationWarning)
                print(f"{ctx.blurb.verbose_status} {img_file_or_folder}' {ctx.blurb.verbose_dated} {args.model}")

            inference_result = {}
            for saved_model, model_data in tqdm(negate_models.items(), disable=args.verbose):
                if isinstance(model_data, str):
                    model_data = Path(model_data)
                context = InferContext(
                    spec=load_spec(saved_model),
                    model_version=model_data,
                    train_metadata=load_metadata(saved_model),
                    label=args.label,
                    file_or_folder_path=img_file_or_folder,
                    dataset_feat=None,
                    run_heuristics=False,
                    model=True,
                    verbose=args.verbose,
                )
                inference_result[saved_model] = infer_origin(context)

            inference_results = (v for _, v in inference_result.items())
            compute_weighted_certainty(
                *inference_results,
                label=args.label,
            )
            # return inferences

        case _:
            raise NotImplementedError


def main():
    """CLI argument parser and command dispatcher.\n
    :raises ValueError: Missing image path.
    :raises ValueError: Invalid VAE choice.
    :raises NotImplementedError: Unsupported command passed.
    """

    spec = Spec()
    blurb = Blurb(spec)
    models_path = root_folder / "models"
    results_path = root_folder / "results"

    inference_pair = ["20260225_185933", "20260225_221149"]  # [FLUX-AE, DC-AE]

    list_results: list[str] = []
    if len(os.listdir(results_path)) > 0:
        list_results = [str(folder.stem) for folder in Path(results_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_results.sort(reverse=True)

    inference_pair = ["20260225_185933", "20260225_221149"]  # [FLUX-AE, DC-AE]

    list_results = []
    if len(os.listdir(results_path)) > 0:
        list_results = [str(folder.stem) for folder in Path(results_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_results.sort(reverse=True)

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help=blurb.pretrain)
    train_parser = subparsers.add_parser("train", help=blurb.train)
    train_parser.add_argument("-l", "--loop", action="store_true", help=blurb.loop)
    train_parser.add_argument("-f", "--features", choices=list_results, default=None, help=blurb.features_load)

    for sub in [pretrain_parser, train_parser]:
        sub.add_argument("gne_path", help=blurb.gne_path, nargs="?", default=None)
        sub.add_argument("-s", "--syn", help=blurb.syn_path, nargs="?", default=None)
        sub.add_argument("-m", "--model", choices=blurb.model_choices, default=blurb.default_vit, help=blurb.vit_model_blurb())
        sub.add_argument("-a", "--ae", choices=blurb.ae_choices, default=blurb.default_vae, help=blurb.ae_model_blurb())

    infer_parser = subparsers.add_parser("infer", help=blurb.infer)
    infer_parser.add_argument("path", help=blurb.unidentified_path)
    if len(os.listdir(models_path)) > 0:
        list_model = [str(folder.stem) for folder in Path(models_path).iterdir() if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)]
        list_model.sort(reverse=True)
        if list_model:
            infer_parser.add_argument("-m", "--model", choices=list_model, default=inference_pair, help=blurb.infer_model_blurb(inference_pair))
    else:
        list_model = None
        infer_parser.add_argument("-m", "--model", choices=None, default=None)

    label_grp = infer_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help=blurb.label_gne)
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help=blurb.label_syn)

    infer_parser.add_argument("-v", "--verbose", action="store_true", help=blurb.verbose)
    args = parser.parse_args(argv[1:])

    cmd_context = CmdContext(args=args, blurb=blurb, spec=spec, results_path=results_path, models_path=models_path, list_model=list_model)

    cmd(cmd_context)


if __name__ == "__main__":
    main()
