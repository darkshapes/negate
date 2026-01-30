# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os

from negate import TrainResult, get_time


def save_model(train_result: TrainResult, file_name: str = "negate_") -> None:
    """Persist a trained model and its PCA transformer.\n
    :param train_result: Training output containing model, PCA and metadata.
    :param file_name: Base name for the files written to the *models* folder.
    :return: None"""

    import numpy as np
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
    from sklearn.decomposition import PCA
    from xgboost import Booster

    model: Booster = train_result.model
    seed: int = train_result.seed
    pca: PCA = train_result.pca
    scale_pos_weight: float = train_result.scale_pos_weight
    timestamp = get_time()
    num_features: int = train_result.feature_matrix.shape[1]

    initial_pca_types = [("input", SkFloatTensorType([None, num_features]))]
    negate_pca_onnx = convert_sklearn(pca, initial_types=initial_pca_types)
    pca_file_name = os.path.join("models", f"{file_name}pca_{timestamp}.onnx")
    onnx.save(negate_pca_onnx, pca_file_name)

    negate_xgb_file_name = os.path.join("models", f"{file_name}_{timestamp}.json")
    model.save_model(negate_xgb_file_name)
    metadata_file = os.path.join("models", f"metadata_{timestamp}.npz")
    np.savez(metadata_file, seed=seed, scale_pos_weight=scale_pos_weight)
    print(f"Models saved to disk. {pca_file_name} {negate_xgb_file_name} {metadata_file}")


def save_to_onnx(train_result: TrainResult, file_name: str = "negate_"):
    """Export the trained XGBoost model to ONNX.\n
    :param train_result: Training output containing the XGBoost model.
    :param file_name: Base name for the ONNX file."""
    import onnx

    from negate.to_onnx import DataType, IOShape, ModelInputFormat, ONNXConverter

    num_features = train_result.feature_matrix.shape[1]
    model = train_result.model
    timestamp = get_time()

    input_shape = IOShape(  # XGBoost expects a 2‑D array [batch, features].
        shape=[-1, num_features],  # <num_features> = model.num_features()
        dtype=DataType.TYPE_FP32,  # XGBoost works with float32
        name="input",
        format=ModelInputFormat.FORMAT_NONE,
    )

    negate_onnx_file_name = os.path.join("models", f"{file_name}_{timestamp}.onnx")
    onnx_model = ONNXConverter.from_xgboost(model, inputs=[input_shape], opset=12)

    onnx.save(onnx_model, negate_onnx_file_name)
