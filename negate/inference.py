# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import numpy as np
import xgboost as xgb
import pickle

# Load artifacts
model = xgb.Booster()
model.load_model("model.xgb")

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

meta = np.load("meta.npz")
scale_pos_weight = meta["scale_pos_weight"]


def predict(features: np.ndarray) -> np.ndarray:
    # Apply the same PCA transformation
    feats_pca = pca.transform(features)
    dmat = xgb.DMatrix(feats_pca)
    return model.predict(dmat)
