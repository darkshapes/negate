# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from datasets import Dataset
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ONNXRuntimeError
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from negate.decompose.wavelet import WaveletAnalyze, WaveletContext
from negate.extract.feature_vae import VAEExtract
from negate.io.config import random_state
from negate.io.datasets import generate_dataset, prepare_dataset
from negate.io.spec import Spec
from negate.metrics.heuristics import model_accuracy, weight_gne_feat, weight_syn_feat


@dataclass
class InferContext:
    """Container for inference dependencies."""

    spec: Spec
    model_version: Path
    train_metadata: dict
    file_or_folder_path: Path
    label: int | None = None
    dataset_feat: Dataset | None = None
    syn_check: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.model_version.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_version}")
        metadata_path = self.model_version / "metadata.json"
        if self.train_metadata is None and metadata_path.exists():
            import json

            with open(metadata_path) as f:
                self.train_metadata = json.load(f)


def run_native(features_dataset: np.ndarray, model_version: Path, parameters: dict) -> np.ndarray:
    """Run inference using XGBoost with PCA pre-processing.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Result array of predicted origins."""

    import xgboost as xgb

    model_file_path_named = model_version / "negate.ubj"

    if not model_file_path_named.exists():
        raise FileNotFoundError(f"Model file not found: {str(model_file_path_named)}. Please run 'train' first to create the model.")
    else:
        model_file_path_named = str(model_file_path_named)

    pca_file_path_named = model_version / "negate_pca.pkl"
    with open(pca_file_path_named, "rb") as pca_file:
        pca = pickle.load(pca_file)

    features_pca = pca.transform(features_dataset)

    model = xgb.Booster(params=parameters)
    model.load_model(model_file_path_named)

    result = model.predict(xgb.DMatrix(features_pca))
    return result


def run_onnx(features_dataset: np.ndarray, model_version: Path, parameters: dict) -> np.ndarray | Any:
    """Run inference using ONNX Runtime with PCA pre-processing.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Result array of predicted origins."""

    model_file_path_named = model_version / "negate.onnx"
    if not model_file_path_named.exists():
        raise FileNotFoundError(f"Model file not found: {str(model_file_path_named)}. Please run 'train' first to create the model.")
    else:
        model_file_path_named = str(model_file_path_named)

    features_dataset = features_dataset.astype(np.float32)
    pca_file_path_named = model_version / "negate_pca.onnx"
    session_pca = ort.InferenceSession(pca_file_path_named)
    input_name_pca = session_pca.get_inputs()[0].name
    features_pca = session_pca.run(None, {input_name_pca: features_dataset})[0]

    input_name = ort.get_available_providers()[0]
    features_model = features_pca.astype(np.float32)  # type: ignore

    session = ort.InferenceSession(model_file_path_named)
    print(f"Model '{model_file_path_named}' loaded.")
    input_name = session.get_inputs()[0].name
    inputs = {input_name: features_dataset.astype(np.float32)}  # noqa
    try:
        result: ort.SparseTensor = session.run(None, {input_name: features_model})[0]  # type: ignore
        print(result)
    except (InvalidArgument, ONNXRuntimeError) as error_log:
        import sys

        print(error_log)
        sys.exit()


def batch_preprocessing(dataset: Dataset, spec: Spec) -> Dataset:
    """Apply wavelet analysis transformations to dataset.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column."""
    print("Beginning preprocessing.")

    kwargs = {}
    kwargs["disable_nullable"] = spec.opt.disable_nullable
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size
    if spec.opt.load_from_cache_file is False:
        kwargs["new_fingerprint"] = str(random_state(spec.train_rounds.max_rnd))
    else:
        kwargs["load_from_cache_file"] = True
        kwargs["keep_in_memory"] = True
        kwargs["new_fingerprint"] = spec.hyper_param.seed

    with VAEExtract(spec) as extractor:  # type: ignore
        dataset = dataset.map(
            extractor.forward,
            remove_columns=["image"],
            desc="Computing wavelets...",
            **kwargs,
        )
    return dataset


def preprocessing(dataset: Dataset, spec: Spec, inference=False) -> Dataset:
    """Apply wavelet analysis transformations to dataset.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column."""
    print("Beginning preprocessing.")

    kwargs = {}
    kwargs["disable_nullable"] = spec.opt.disable_nullable
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size
    if spec.opt.load_from_cache_file is False:
        kwargs["new_fingerprint"] = str(random_state(spec.train_rounds.max_rnd))
    else:
        kwargs["load_from_cache_file"] = True
        kwargs["keep_in_memory"] = True
        kwargs["new_fingerprint"] = spec.hyper_param.seed

    context = WaveletContext(spec=spec, inference=inference)
    with WaveletAnalyze(context) as analyzer:  # type: ignore
        dataset = dataset.map(
            analyzer,
            remove_columns=["image"],
            desc="Computing wavelets...",
            **kwargs,
        )
    return dataset


def predict_gne_or_syn(context: InferContext) -> np.ndarray:
    """Returns 0 for GNE-like results, 1 for SYN-like results determined by decision tree model trained on dataset:\n
    :param data_path: Path to json file with saved parameter data"""
    spec = context.spec
    model_version = context.model_version
    assert isinstance(context.dataset_feat, Dataset), ValueError("Dataset was not passed to prediction")
    features_matrix = prepare_dataset(context.dataset_feat, spec)
    parameters = asdict(spec.hyper_param) | {"scale_pos_weight": context.train_metadata["scale_pos_weight"]}
    result = run_onnx(features_matrix, model_version, parameters) if spec.opt.load_onnx else run_native(features_matrix, model_version, parameters=parameters)
    return result


def infer_origin(context: InferContext) -> dict[str, list[tuple[str, int]]]:
    """Predict synthetic or original for given image.\n
    :param context: Inference context containing spec, model path, and metadata.
    :param file_or_folder_path: Path to the image or folder to be checked.
    :param features_dataset: Optional dataset. If not provided, will be computed from input in context.
    :return: Prediction arrays (0=genuine, 1=synthetic)."""

    if context.dataset_feat is None:  # Support lazy loading of features if not provided
        assert context.file_or_folder_path, ValueError("Image path must be provided for inference")
        origin_ds: Dataset = generate_dataset(context.file_or_folder_path)
        context.dataset_feat = preprocessing(origin_ds, context.spec)
    model_pred = predict_gne_or_syn(context=context)
    model_pred = model_accuracy(model_pred)
    heur_dc_pred = []
    heur_ae_pred = []
    for entry in context.dataset_feat["results"]:
        heur_dc_pred.append(weight_gne_feat(entry[0]))
        heur_ae_pred.append(weight_syn_feat(entry[0]))

    if context.verbose:
        print(f"""          Decision Tree Model result: {model_pred}
            SYN Probability (DC VAE): {heur_dc_pred}
            GNE Probability (AE VAE): {heur_ae_pred}""")
    return {"unk": model_pred, "syn": heur_dc_pred, "gne": heur_ae_pred}
