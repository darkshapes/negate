# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from datetime import datetime

import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
from xgboost import Booster

from negate import TrainResult

get_time = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model(train_result: TrainResult, file_name: str = "negate_") -> None:
    """Persist a trained model and its PCA transformer.\n
    :param train_result: Training output containing model, PCA and metadata.
    :param file_name: Base name for the files written to the *models* folder.
    :return: None"""
    model: Booster = train_result.model
    feature_matrix = train_result.feature_matrix
    seed = train_result.seed
    pca = train_result.pca
    scale_pos_weight = train_result.scale_pos_weight
    time_data = get_time()
    num_features = train_result.num_features

    initial_pca_types = [("input", SkFloatTensorType([None, feature_matrix.shape[1]]))]
    negate_pca_onnx = convert_sklearn(pca, initial_types=initial_pca_types)
    pca_file_name = os.path.join("models", f"{file_name}pca_{time_data}.onnx")
    onnx.save(negate_pca_onnx, pca_file_name)

    negate_xgb_file_name = os.path.join("models", f"{file_name}_{time_data}.json")
    model.save_model(negate_xgb_file_name)
    metadata_file = os.path.join("models", f"metadata_{time_data}.npz")
    np.savez(metadata_file, seed=seed, scale_pos_weight=scale_pos_weight)
    print(f"Models saved to disk. {pca_file_name} {negate_xgb_file_name} {metadata_file}")


def save_to_onnx(train_result: TrainResult, file_name: str = "negate_"):
    """Export the trained XGBoost model to ONNX.\n
    :param train_result: Training output containing the XGBoost model.
    :param file_name: Base name for the ONNX file."""

    from negate.to_onnx import ONNXConverter
    from negate.conversion import IOShape, ModelInputFormat, DataType

    num_features = train_result.feature_matrix.shape[1]
    model = train_result.model
    time_data = get_time()

    input_shape = IOShape(  # XGBoost expects a 2‑D array [batch, features].
        shape=[-1, num_features],  # <num_features> = model.num_features()
        dtype=DataType.TYPE_FP32,  # XGBoost works with float32
        name="input",
        format=ModelInputFormat.FORMAT_NONE,
    )

    negate_onnx_file_name = os.path.join("models", f"{file_name}_{time_data}.onnx")
    onnx_model = ONNXConverter.from_xgboost(model, inputs=[input_shape], opset=12)

    import onnx

    onnx.save(onnx_model, negate_onnx_file_name)
