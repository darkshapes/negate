# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""CLI command implementations for the Negate package."""

from __future__ import annotations

from typing import Any
import time as timer_module

start_ns = timer_module.perf_counter()


def cmd(ctx: Any) -> None:
    """Execute CLI command based on parsed arguments.\n
    :param ctx: Command context with parsed args and runtime dependencies.\n
    """

    args = ctx.args

    configure_runtime_logging()

    match args.cmd:
        case "pretrain":
            from negate.io.save import end_processing, save_features
            from negate.metrics.track import chart_decompositions
            from negate.train import build_train_call, pretrain

            origin_ds = build_train_call(args=args, path_result=ctx.results_path, spec=ctx.spec)
            features_ds = pretrain(origin_ds, ctx.spec)
            end_processing("Pretraining", start_ns)
            save_features(features_ds)
            chart_decompositions(features_dataset=features_ds, spec=ctx.spec)

        case "train":
            from negate.io.save import end_processing, save_train_result
            from negate.metrics.track import run_training_statistics
            from negate.train import build_train_call, train_model, training_loop

            origin_ds = build_train_call(args=args, path_result=ctx.results_path, spec=ctx.spec)
            if args.loop is True:
                training_loop(image_ds=origin_ds, spec=ctx.spec)
            else:
                train_result = train_model(features_ds=origin_ds, spec=ctx.spec)
                timecode = end_processing("Training", start_ns)
                save_train_result(train_result)
                run_training_statistics(train_result=train_result, timecode=timecode, spec=ctx.spec)

        case "infer":
            from tqdm import tqdm

            from negate.inference import InferContext, infer_origin, preprocessing
            from negate.io.datasets import generate_dataset
            from negate.io.spec import load_metadata
            from negate.metrics.heuristics import compute_weighted_certainty

            if args.path is None:
                raise ValueError(ctx.blurb.infer_path_error)
            if ctx.list_model is None or not ctx.list_model:
                raise ValueError(f"{ctx.blurb.model_error} {ctx.models_path} {ctx.blurb.model_error_hint}")

            img_file_or_folder = Path(args.path)
            if not isinstance(args.model, list) and not isinstance(args.model, tuple):
                raise ValueError(ctx.blurb.model_pair)

            negate_models: dict[str, Path] = {}
            model_specs: dict[str, Any] = {}
            model_metadata: dict[str, Any] = {}
            for saved_model in args.model:
                negate_models[saved_model] = ctx.models_path / saved_model
                if not negate_models[saved_model].exists():
                    raise ValueError(ctx.blurb.model_pattern)
                model_specs[saved_model] = load_spec(saved_model)
                model_metadata[saved_model] = load_metadata(saved_model)

            if args.verbose:
                import warnings

                warnings.filterwarnings("default", category=UserWarning)
                warnings.filterwarnings("default", category=DeprecationWarning)
                CLI_LOGGER.info(f"{ctx.blurb.verbose_status} {img_file_or_folder}' {ctx.blurb.verbose_dated} {args.model}")

            CLI_LOGGER.info("Preparing feature dataset and loading selected models...")
            origin_ds = generate_dataset(img_file_or_folder, verbose=args.verbose)
            feature_cache: dict[str, Any] = {}
            feature_key_by_model: dict[str, str] = {}
            for saved_model, model_spec in model_specs.items():
                feature_key = "|".join(
                    [
                        str(model_spec.model),
                        str(model_spec.vae),
                        str(model_spec.dtype),
                        str(model_spec.device),
                        str(model_spec.opt.dim_factor),
                        str(model_spec.opt.dim_patch),
                        str(model_spec.opt.top_k),
                        str(model_spec.opt.condense_factor),
                        str(model_spec.opt.alpha),
                        str(model_spec.opt.magnitude_sampling),
                    ]
                )
                feature_key_by_model[saved_model] = feature_key
                if feature_key not in feature_cache:
                    feature_cache[feature_key] = preprocessing(origin_ds, model_spec, verbose=args.verbose)

            inference_result = {}
            for saved_model, model_data in tqdm(
                negate_models.items(),
                total=len(negate_models),
                desc="Running inference with each selected model",
                disable=False,
            ):
                context = InferContext(
                    spec=model_specs[saved_model],
                    model_version=model_data,
                    train_metadata=model_metadata[saved_model],
                    label=args.label,
                    file_or_folder_path=img_file_or_folder,
                    dataset_feat=feature_cache[feature_key_by_model[saved_model]],
                    run_heuristics=False,
                    model=True,
                    verbose=args.verbose,
                )
                inference_result[saved_model] = infer_origin(context)

            inference_results = (result for _, result in inference_result.items())
            compute_weighted_certainty(*inference_results, label=args.label)

        case "process":
            from negate.extract.combination import run_all_combinations
            from negate.extract.unified_core import ExtractionModule, UnifiedExtractor
            from negate.io.spec import Spec
            from PIL import Image

            img_file_or_folder = Path(args.path)
            spec = Spec()
            all_modules = list(ExtractionModule)

            transposed = args.transposed
            if transposed is not None:
                try:
                    transposed = [int(x) for x in transposed.split(",")]
                except ValueError:
                    print("Error: transposed must be comma-separated integers")
                    exit(1)

            combo = args.combination
            if combo is None:
                combo = [mod.name for mod in all_modules]

            CLI_LOGGER.info(f"Running process on {img_file_or_folder}...")
            CLI_LOGGER.info(f"Transposed: {transposed}")
            CLI_LOGGER.info(f"Combination: {combo}")

            results: dict[str, Any] = {"transposed": {}, "combination": {}}

            if transposed:
                for idx in transposed:
                    if idx >= len(all_modules):
                        print(f"Error: transposed index {idx} out of range")
                        exit(1)
                    try:
                        extractor = UnifiedExtractor(spec, enable=[all_modules[idx]])
                        features = extractor(Image.open(img_file_or_folder).convert("RGB"))
                        results["transposed"][all_modules[idx].name] = features
                        extractor.cleanup()
                    except Exception as e:
                        results["transposed"][all_modules[idx].name] = {}
                        CLI_LOGGER.warning(f"Error processing module {all_modules[idx].name}: {e}")

            for mod_name in combo:
                if mod_name not in all_modules:
                    print(f"Error: combination module {mod_name} not found")
                    exit(1)
                try:
                    extractor = UnifiedExtractor(spec, enable=[ExtractionModule[mod_name]])
                    features = extractor(Image.open(img_file_or_folder).convert("RGB"))
                    results["combination"][mod_name] = features
                    extractor.cleanup()
                except Exception as e:
                    results["combination"][mod_name] = {}
                    CLI_LOGGER.warning(f"Error processing module {mod_name}: {e}")

            output_file = ctx.results_path / "process_results.json"
            import json

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            CLI_LOGGER.info(f"Results saved to {output_file}")

            if args.train:
                from negate.io.spec import load_metadata
                from negate.train import train_model, build_train_call, save_train_result
                from negate.metrics.track import run_training_statistics
                from negate.io.save import end_processing

                model_path = ctx.results_path / "process_results.json"
                if not model_path.exists():
                    print(f"Error: No results found at {model_path}")
                    exit(1)

                model_spec = {
                    "model": "convnext" if args.train == "convnext" else "xgboost",
                    "vae": "",
                    "dtype": "float64",
                    "device": "cpu",
                    "opt": {
                        "dim_factor": 3,
                        "dim_patch": 16,
                        "top_k": 20,
                        "condense_factor": 2,
                        "alpha": 0.01,
                        "magnitude_sampling": "top_k",
                    },
                }

                train_result = train_model(features_ds=results, spec=spec)
                timecode = end_processing(f"Training ({args.train})", start_ns)
                save_train_result(train_result)
                run_training_statistics(train_result=train_result, timecode=timecode, spec=spec)
                CLI_LOGGER.info(f"Training ({args.train}) completed with accuracy: {train_result.get('accuracy', 0.0):.4f}")

        case _:
            raise NotImplementedError


if __name__ == "__main__":
    from negate.io.spec import Spec

    spec = Spec()
    blurb = Blurb(spec)
    cmd(
        CmdContext(
            args=None,
            blurb=blurb,
            spec=spec,
            results_path=None,
            models_path=None,
            list_model=None,
        )
    )
