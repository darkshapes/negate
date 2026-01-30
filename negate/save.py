# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import numpy as np
import onnx
from onnxmltools.convert.xgboost.convert import convert as convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
from xgboost import Booster
from datetime import datetime
from negate import TrainResult
import os


def save_model(train_result: TrainResult, file_name: str = "negate_", extension: str = ".onnx") -> None:
    model: Booster = train_result.model
    feature_matrix = train_result.feature_matrix
    seed = train_result.seed
    pca = train_result.pca
    time_data = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_meta_data = datetime.now().isoformat()

    initial_pca_types = [("input", SkFloatTensorType([None, feature_matrix.shape[1]]))]
    negate_pca_onnx = convert_sklearn(pca, initial_types=initial_pca_types)
    pca_file_name = os.path.join("models", f"{file_name}pca_{time_data}{extension}")
    onnx.save(negate_pca_onnx, pca_file_name)

    negate_xgb = f"{file_name}_{time_data}.xgb"
    model.save_model(negate_xgb)

    metadata_file = os.path.join("models", f"metadata_{time_data}.npz")
    np.savez(metadata_file, seed=seed)
    print(f"Models saved to disk. {pca_file_name} {negate_xgb} {metadata_file}")

    num_features: int = model.num_features
    feature_names = model.feature_names
    if feature_names:
        # ONNX pattern (f0, f1, f2, ...)
        if needs_fixing := any(not name.startswith("f") or not name[1:].isdigit() for name in feature_names if name):
            print(f"Converting feature names : {needs_fixing}")
            model.feature_names = [f"f{feature}" for feature in range(num_features)]

    # print(feature_matrix.shape[1])
    # print(num_features)
    # initial_xgb_type = [("input", OnnxFloatTensorType([None, num_features]))]
    # negate_xgb_onnx = f"{file_name}xgb{extension}"

    # try:
    #     onnx_model = convert_xgboost(
    #         model,
    #         initial_types=initial_xgb_type,
    #         target_opset=15,
    #         doc_string=f"XGBoost AI image detection - {str(seed)} - {str(time_meta_data)}",
    #     )
    #     onnx.save(onnx_model, negate_xgb_onnx)
    # except ValueError as e:
    #     if "Unsupported dimension type" in str(e):
    #         print("Error: Unsupported dimension ")
    #         if feature_matrix.shape[1] != num_features:
    #             print("Feature matrix dimension mismatch.")
    #             model.feature_names = [f"f{i}" for i in range(feature_matrix.shape[1])]
    #         else:
    #             print("Input shape matches expected format...")
    #     else:
    #         raise e
