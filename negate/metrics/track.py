# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import iqr
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from negate.io.config import Spec
from negate.metrics.plot import (
    graph_cohen,
    graph_kde,
    graph_residual,
    graph_tail_separations,
    graph_vae_loss,
    graph_wavelet,
    residual_keys,
    result_path,
    vae_loss_keys,
    wavelet_keys,
    graph_train_variance,
)
from negate.train import TrainResult, timestamp


def accuracy(train_result: TrainResult, timecode: float):
    """Print diagnostics and plots for a trained model.\n
    :param train_result: Result object from training.
    :param timecode: Elapsed time since launch"""

    model = train_result.model
    d_matrix_test = train_result.d_matrix_test
    feature_matrix = train_result.feature_matrix
    labels = train_result.labels
    pca = train_result.pca
    seed = train_result.seed
    y_test = train_result.y_test
    X_train = train_result.X_train
    X_train_pca = train_result.X_train_pca
    scale_pos_weight = train_result.scale_pos_weight

    y_pred_proba = model.predict(d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    results = {
        "accuracy": accuracy,
        "best_iter": model.best_iteration,
        "best_score": model.best_score,
        "cumulative:": np.cumsum(pca.explained_variance_ratio_),
        "explained_var": pca.explained_variance_ratio_.sum(),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "feature_shape": feature_matrix.shape,
        "label_dist": {
            "real": int(np.sum(labels == 0)),
            "synthetic": int(np.sum(labels == 1)),
            "real_pct": np.sum(labels == 0) / len(labels) * 100,
            "synthetic_pct": np.sum(labels == 1) / len(labels) * 100,
        },
        "imbalance_ratio": np.sum(labels == 0) / np.sum(labels == 1),
        "labels_shape": labels.shape,
        "n_components": pca.n_components_,  # type: ignore[attr-defined]
        "original_dim": X_train.shape[1],
        "pca_dim": X_train_pca.shape[1],
        "roc_auc": roc_auc,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "time_elapsed": timecode,
        "timestamp": timestamp,
    }

    result_path.mkdir(parents=True, exist_ok=True)
    results_file = str(result_path / f"results_{timestamp}.json")
    result_format = {k: str(v) for k, v in results.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)
    separator = lambda: print("=" * 60)
    separator()
    print("CLASSIFICATION RESULTS")
    separator()
    print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))


def compute_tail_separation(data_frame, residual_keys) -> pd.DataFrame:
    """Score metrics by class separation and long-tail outlier tendency.\n
    :param data_frame: Expanded DataFrame with residual columns.
    :param residual_keys: List of metric column names to analyze.
    :returns: DataFrame with scores sorted by separateness."""

    results = []
    for key in residual_keys:
        vals_0 = data_frame[data_frame["label"] == 0][key].dropna().values
        vals_1 = data_frame[data_frame["label"] == 1][key].dropna().values

        if len(vals_0) < 5 or len(vals_1) < 5:
            continue

        def tail_score(arr):
            q75, q25 = np.percentile(arr, [75, 25])
            iqr_ = q75 - q25
            if iqr_ == 0:
                return 0
            return (np.percentile(arr, 95) - np.median(arr)) / iqr_

        tail_0 = tail_score(vals_0)
        tail_1 = tail_score(vals_1)
        avg_tail = (tail_0 + tail_1) / 2

        p95_0, p95_1 = np.percentile(vals_0, 95), np.percentile(vals_1, 95)
        p05_0, p05_1 = np.percentile(vals_0, 5), np.percentile(vals_1, 5)

        overlap_gid = max(p05_0, p05_1) - min(p95_0, p95_1)
        pooled_iqr = (iqr(vals_0) + iqr(vals_1)) / 2

        separation_norm = abs((np.median(vals_0) - np.median(vals_1))) / (pooled_iqr if pooled_iqr > 0 else 1)
        combined_score = avg_tail * max(0, separation_norm)

        results.append({"metric": key, "tail_score": avg_tail, "separation": separation_norm, "combined_score": combined_score, "no_overlap": overlap_gid < 0})

    return pd.DataFrame(results).sort_values("combined_score", ascending=False)


def chart_decompositions(features_dataset: Dataset, spec: Spec) -> None:
    """Plot wavelet sensitivity distributions."""

    data_frame = features_dataset.to_pandas()
    expanded_frame = data_frame.explode("results").reset_index(drop=True)  # type: ignore explode

    total_keys = wavelet_keys + residual_keys + vae_loss_keys
    for key in total_keys:
        if key in expanded_frame:
            expanded_frame[key] = expanded_frame["results"].apply(lambda x, k=key: float(np.mean(x[k])) if isinstance(x, dict) and k in x else None)

    scores_dataframe = compute_tail_separation(expanded_frame, residual_keys)
    graph_tail_separations(spec, scores_dataframe=scores_dataframe)
    graph_wavelet(spec, wavelet_dataframe=expanded_frame)
    graph_residual(spec, residual_dataframe=expanded_frame)
    graph_kde(spec, residual_dataframe=expanded_frame)
    graph_cohen(spec, residual_dataframe=expanded_frame)
    graph_vae_loss(spec, vae_dataframe=expanded_frame)
    print(f"[TRACK] Saved plots to {result_path}")


def run_feature_statistics(features_dataset: Dataset, spec: Spec):
    from negate.io.save import save_features

    json_path = save_features(features_dataset)
    chart_decompositions(features_dataset=features_dataset, spec=spec)
    return json_path


def run_training_statistics(train_result: TrainResult, timecode: float, spec: Spec):
    accuracy(train_result=train_result, timecode=timecode)
    graph_train_variance(train_result=train_result, spec=spec)
