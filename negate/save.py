# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pickle

import numpy as np

from negate import TrainResult


def save_metadata(train_result: TrainResult, file_name: str = "negate") -> str:
    """Save training metadata.\n
    :param train_result: Training output containing scale_pos_weight and seed.
    :param file_name: Base file name for the metadata file.
    :return: Path to the saved metadata file."""
    from negate.train import generate_datestamp_path

    scale_pos_weight: float = train_result.scale_pos_weight  # type: ignore
    seed: int = train_result.seed

    metadata_file_name = generate_datestamp_path(f"{file_name}_metadata.npz")
    np.savez(metadata_file_name, seed=seed, scale_pos_weight=scale_pos_weight)
    return metadata_file_name


def save_models(train_result: TrainResult, compare: bool, file_name: str = "negate") -> None:
    """Persist a trained model and its PCA transformer.\n
    :param train_result: Training output containing model, PCA and metadata.
    :param file_name: Base name for the files written to the *models* folder.
    :return: None"""
    from negate.train import generate_datestamp_path

    try:
        from sklearn.decomposition import PCA
        from xgboost import Booster
    except (ImportError, ModuleNotFoundError, Exception):
        raise RuntimeError("missing dependencies for onnx. Please install it using 'negate[onnx]'")
    datestamp_path = generate_datestamp_path(file_name)

    model: Booster = train_result.model
    pca: PCA = train_result.pca
    pca_file_name = datestamp_path + "_pca.pkl"
    with open(pca_file_name, "wb") as f:
        pickle.dump(pca, f)

    negate_xgb_file_name = datestamp_path + ".ubj"
    model.save_model(negate_xgb_file_name)

    metadata_file_name = save_metadata(train_result)

    print(f"Models saved to disk. {pca_file_name} {negate_xgb_file_name} {metadata_file_name}")


def save_to_onnx(train_result: TrainResult, file_name: str = "negate"):
    """Export the trained XGBoost model to ONNX.\n
    :param train_result: Training output containing the XGBoost model.
    :param file_name: Base name for the ONNX file."""

    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except (ImportError, ModuleNotFoundError, Exception):
        raise RuntimeError("missing dependencies for onnx. Please install it using 'negate[onnx]'")
    from negate.train import generate_datestamp_path
    from negate.to_onnx import DataType, IOShape, ModelInputFormat, ONNXConverter

    datestamp_path = generate_datestamp_path(file_name)

    model = train_result.model
    num_features = train_result.feature_matrix.shape[1]
    pca = train_result.pca

    input_shape = IOShape(  # XGBoost expects a 2‑D array [batch, features] at float32
        shape=[-1, num_features],
        dtype=DataType.TYPE_FP32,
        name="input",
        format=ModelInputFormat.FORMAT_NONE,  # Used for TensorRT
    )
    negate_onnx_file_name = datestamp_path + ".onnx"
    onnx_model = ONNXConverter.from_xgboost(model, inputs=[input_shape], opset=12)
    onnx.save(onnx_model, negate_onnx_file_name)

    initial_pca_types = [("input", FloatTensorType([None, num_features]))]
    negate_pca_onnx_raw = convert_sklearn(pca, initial_types=initial_pca_types)
    negate_pca_onnx = ONNXConverter.optim_onnx(negate_pca_onnx_raw)  # type: ignore[arg-type]
    pca_file_name = datestamp_path + "_pca.onnx"
    onnx.save(negate_pca_onnx, pca_file_name)

    metadata_file_name = save_metadata(train_result)

    print(f"Models saved to disk. {pca_file_name} {negate_onnx_file_name} {metadata_file_name}")
