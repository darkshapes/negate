# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Generate results PDF with multi-signal ensemble, calibrated thresholds,
abstention, and full precision/recall/F1 reporting.

Usage: uv run python tests/generate_results_pdf.py
Output: results/artwork_detection_results.pdf
"""

from __future__ import annotations

from typing import Any

from negate.decompose.surface import SurfaceFeatures
from negate.io.datasets import build_datasets
from negate.io.spec import Spec
from negate.metrics.pdf import generate_pdf


def load_and_extract(spec: Spec) -> tuple[Any, Any, list[str], Any, Any]:
    """Load dataset and extract surface features for ensemble evaluation.\n
    :param spec: Specification containing data paths and hyperparameters.\n
    :returns: Tuple of (features array, labels, feature names, gen images, synthetic images).\n
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    genuine_repo = spec.data.genuine_data[0] if spec.data.genuine_data else None
    synthetic_repo = spec.data.synthetic_data[0] if spec.data.synthetic_data else None
    sample_size = spec.ensemble.sample_size

    print(f"Loading {sample_size} human art + {sample_size} AI images...")

    dataset = build_datasets(spec, genuine_repo, synthetic_repo)
    extractor = SurfaceFeatures
    features: list[dict[str, float]] = []
    labels: list[int] = []

    for row in tqdm(dataset, desc="Extracting artwork features"):
        features.append(extractor(row["image"]))  # type: ignore
        labels.append(row["label"])  # type: ignore

    df = pd.DataFrame(features).fillna(0)
    X = np.where(np.isfinite(df.to_numpy(dtype=np.float64)), df.to_numpy(dtype=np.float64), 0)
    y = np.array(labels)
    gen_data = dataset.filter(lambda x: x["label"] == 0)
    syn_data = dataset.filter(lambda x: x["label"] == 1)
    return X, y, list(df.columns), gen_data, syn_data


def run_ensemble_cv(X: Any, y: Any, spec: Spec) -> tuple[dict[str, Any], Any, Any, Any]:
    """Run calibrated ensemble with abstention using spec hyperparameters.\n
    :param X: Feature matrix.\n
    :param y: Label vector.\n
    :param spec: Specification containing model hyperparameters and config.\n
    :returns: Tuple of (results dict, ensemble probabilities, predictions, full model).\n
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    hp = spec.hyper_param
    ens = spec.ensemble

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=ens.n_folds, shuffle=True, random_state=hp.seed)

    models = {
        "SVM": CalibratedClassifierCV(SVC(C=ens.svm_c, gamma=ens.gamma, kernel=ens.kernel, random_state=hp.seed), cv=ens.cv, method=ens.method),
        "MLP": CalibratedClassifierCV(
            MLPClassifier(hidden_layer_sizes=(ens.mlp_hidden_layers,), activation=ens.mlp_activation, max_iter=ens.mlp_max_iter, random_state=hp.seed),
            cv=ens.cv,
            method=ens.method,
        ),
    }

    model_probs = {}
    model_preds = {}
    for name, model in models.items():
        probs = cross_val_predict(model, X_s, y, cv=skf, method="predict_proba")[:, 1]  # type: ignore
        model_probs[name] = probs
        model_preds[name] = int(probs > 0.5)

    xgb_probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X_s, y):
        params = {
            "sample_size": ens.sample_size,
            "abstain_threshold": ens.abstain_threshold,
            "n_folds": ens.n_folds,
            **hp,  # type: ignore
        }
        dtrain = xgb.DMatrix(X_s[train_idx], label=y[train_idx])
        dtest = xgb.DMatrix(X_s[test_idx])
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=spec.train_rounds.num_boost_round,
            evals=[(xgb.DMatrix(X_s[test_idx], label=y[test_idx]), "test")],
            early_stopping_rounds=spec.train_rounds.early_stopping_rounds,
            verbose_eval=spec.train_rounds.verbose_eval,
        )
        xgb_probs[test_idx] = model.predict(dtest)

    model_probs["XGBoost"] = xgb_probs
    model_preds["XGBoost"] = np.where(xgb_probs > 0.5, 1, 0)

    ensemble_probs = sum(model_probs.values()) / len(model_probs)
    ensemble_preds = np.where(ensemble_probs > 0.5, 1, 0)

    model_probs["Ensemble"] = ensemble_probs
    model_preds["Ensemble"] = ensemble_preds

    results = {}
    for name in model_probs:
        probs = model_probs[name]
        preds = model_preds[name]
        results[name] = {
            "accuracy": np.mean(preds == y),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
        }

    abstain_thresh = ens.abstain_threshold
    uncertain_mask = (ensemble_probs > abstain_thresh) & (ensemble_probs < (1 - abstain_thresh))
    confident_preds = ensemble_preds.copy()
    confident_preds[uncertain_mask] = -1  # Mark uncertain as -1

    results["Ensemble_With_Abstention"] = {
        "accuracy": np.sum(confident_preds == y) / (y.shape[0] - np.sum(uncertain_mask)) if (y.shape[0] - np.sum(uncertain_mask)) > 0 else 0,
        "abstention_rate": np.mean(uncertain_mask),
    }

    full_xgb_params = {**spec.hyper_param}  # type: ignore
    full_model = xgb.train(full_xgb_params, xgb.DMatrix(X_s, label=y), num_boost_round=spec.train_rounds.num_boost_round)

    return results, ensemble_probs, ensemble_preds, full_model


def main():
    import numpy as np

    X, y, names, imgs_h, imgs_a = load_and_extract()  # type: ignore
    print(f"Dataset: {np.sum(y == 0)} Genuine + {np.sum(y == 1)} Synthetic, {X.shape[1]} features")

    results, ens_probs, ens_preds, model = run_ensemble_cv(X, y, None)  # type: ignore

    print(f"\n{'Model':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 55)
    for name, r in results.items():
        extra = f"  ({r.get('n_abstained', '-')} abstained)" if "n_abstained" in r else ""
        print(f"{name:<15} {r['accuracy']:>7.1%} {r['precision']:>7.1%} {r['recall']:>7.1%} {r['f1']:>7.1%} {r['roc_auc']:>7.4f}{extra}")

    generate_pdf(X, y, names, results, ens_probs, ens_preds, model, imgs_h, imgs_a)
    print("Done.")


if __name__ == "__main__":
    main()
