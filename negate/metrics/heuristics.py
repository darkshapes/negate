# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from datasets import Dataset
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ONNXRuntimeError
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from negate.io.config import Spec
from negate.train import prepare_dataset


@dataclass
class InferContext:
    """Container for inference dependencies."""

    spec: Spec
    model_version: Path
    train_metadata: dict
    label: bool | None = None

    def __post_init__(self) -> None:
        if not self.model_version.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_version}")
        metadata_path = self.model_version / "metadata.json"
        if self.train_metadata is None and metadata_path.exists():
            import json

            with open(metadata_path) as f:
                self.train_metadata = json.load(f)


def weigh_bce(entry):
    bce_threshold = -80  # SYN has very negative bce_loss (< -80 typically)
    ff_threshold = 0.0  # SYN tends to be higher in mean ff magnitude
    min_base_threshold = 100

    bce_loss = entry["results"][0]["bce_loss"]  # bce_loss is much more negative in SYN (around -100+ vs GNE around -50)
    image_mean_ff = entry["results"][0]["image_mean_ff"]  # Another indicator: Mean is dominated by negatives for SYN
    min_base = entry["results"][0]["min_base"]  # min_base heuristic: GNE clusters around 1000-1300 range more tightly
    score = 0
    confidence = 1

    if bce_loss is not None:
        if bce_loss < -150 or (-15 < bce_loss < -5):
            score += 1  # More likely SYN
        if bce_loss > -20 or bce_loss < -170:
            score += 1
        if bce_threshold < bce_loss < -50 and min_base_threshold < min_base < 1400:
            score -= 1
        confidence += 1

    if image_mean_ff > ff_threshold:
        score += 1
    elif bce_loss is not None and -100 > bce_loss > -50:
        score -= 1  # More likely GNE

    if min_base is not None:
        if min_base_threshold <= min_base <= 1350:
            score -= 1  # More likely GNE (tighter cluster)
        if min_base > 4000 or min_base < 200:
            score += 1
        confidence += 1
    return score, confidence


def weigh_v2(entry):
    # flux2 klein fp16, timm/vit-base dino v3 lvd
    #  dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5                  # strength of perturbation Default 0.5)

    laplace = entry["laplace_mean"]  # Above ~4.2-4.3 for GNE
    sobel = entry["sobel_mean"]  # Below 4 for GNE
    max_ff_mag = entry["max_fourier_magnitude"]  # Above 1500 for GNE
    bce_loss = entry["bce_loss"]  # bce_los GNE around -50 -60 is more likely to be GNE)

    score = 2
    confidence = 0
    if laplace is not None and any(x > 4.2 for x in laplace):
        confidence += 1
        score -= 1

    if sobel is not None and any(x < 4 for x in sobel):
        confidence += 1
        score -= 1

    if max_ff_mag is not None and max_ff_mag > 1500:
        confidence += 1
        score -= 1

    if -50 > bce_loss > -60:
        confidence += 1
        score += 0.6

    return score, confidence


def classify_gnf_or_syn(dataset: Dataset, label: int) -> list[int]:
    """Returns 0 for GNE-like results, 1 for SYN-like results determined by simple heuristic based on observed patterns:\n
    :param data_path: Path to json file with saved parameter data"""

    result = {}
    entries = dataset["results"]

    # Collect ground truth from entry if available
    gt_labels = []
    predictions = []

    for index, entry in enumerate(entries):
        score, confidence = weigh_v2(entry[0])
        class_pred = "SYN" if score > 0 else "GNE"
        result[str(index)] = {"score": score, "class": class_pred, "confidence": confidence}

        predictions.append(1 if class_pred == "SYN" else 0)

        if label is not None:
            gt_labels.append(label)  # or however you access ground truth

    if label is not None:
        acc = float(np.mean([p == g for p, g in zip(predictions, gt_labels)]))
        print(f"Accuracy: {acc:.2%}")
    print(predictions)
    return predictions


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
    features_model = features_pca.astype(np.float32)

    session = ort.InferenceSession(model_file_path_named)
    print(f"Model '{model_file_path_named}' loaded.")
    input_name = session.get_inputs()[0].name
    inputs = {input_name: features_dataset.astype(np.float32)}
    try:
        result: ort.SparseTensor = session.run(None, {input_name: features_model})[0]  # type: ignore
        print(result)
    except (InvalidArgument, ONNXRuntimeError) as error_log:
        import sys

        print(error_log)
        sys.exit()

    # session_pca = ort.InferenceSession(pca_file_path_named)
    # input_name_pca = session_pca.get_inputs()[0].name
    # features_pca = session_pca.run(None, {input_name_pca: features_dataset})[0]

    # input_name = ort.get_available_providers()[0]
    # pca_file_path_named = model_version / "negate_pca.pkl"
    # with open(pca_file_path_named, "rb") as pca_file:
    #     pca = pickle.load(pca_file)

    # features_pca = pca.transform(features_dataset)
    # features_model = features_pca.astype(np.float32)  # type: ignore
    # session_pca = ort.InferenceSession("negate_pca.onnx")
    # input_name_pca = session_pca.get_inputs()[0].name
    # features_pca = session_pca.run(None, {input_name_pca: np.array(features_dataset).astype(np.float32)})[0]

    # input_name = ort.get_available_providers()[0]
    # inputs = {input_name: features_dataset.astype(np.float32)}

    # session = ort.InferenceSession(model_file_path_named)
    # print(f"Model '{model_file_path_named}' loaded.")
    # return session.run(None, inputs)[0]

    # session = ort.InferenceSession(model_file_path_named)
    # print(f"Model '{model_file_path_named}' loaded.")
    # input_name = session.get_inputs()[0].name
    # try:
    #     result = session.run(None, {input_name: features_model})[0]  # type: ignore
    #     print(result)
    #     return result
    # except (InvalidArgument, ONNXRuntimeError) as error_log:
    #     import sys

    #     print(error_log)
    #     sys.exit()


def infer_origin(context: InferContext, features_dataset: Dataset | None = None) -> np.ndarray:
    """Predict synthetic or original for given image.\n
    :param context: Inference context containing spec, model path, and metadata.
    :param features_dataset: Optional dataset. If not provided, will be computed from input in context.
    :return: Prediction arrays (0=genuine, 1=synthetic)."""

    if features_dataset is None:  # Support lazy loading of features if not provided
        raise ValueError("features_dataset must be provided or infer_origin called with full context")

    spec = context.spec
    model_version = context.model_version

    features_matrix = prepare_dataset(features_dataset, spec)
    parameters = asdict(spec.hyper_param) | {"scale_pos_weight": context.train_metadata["scale_pos_weight"]}
    result = run_onnx(features_matrix, model_version, parameters) if spec.opt.load_onnx else run_native(features_matrix, model_version, parameters=parameters)

    thresh = 0.5
    predictions = (result > thresh).astype(int)
    if context.label is not None:
        ground_truth = np.full(predictions.shape, context.label, dtype=int)
        acc = float(np.mean(predictions == ground_truth))
        print(f"Accuracy: {acc:.2%}")
    print(result)
    print(predictions)
    return predictions
