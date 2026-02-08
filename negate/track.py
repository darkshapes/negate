# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from negate.train import TrainResult

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = Path(__file__).parent.parent / "results" / timestamp


def accuracy(train_result: TrainResult):
    """Print diagnostics and plots for a trained model.

    :param train_result: Result object from training."""

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
        "timestamp": timestamp,
    }

    results_file = str(result_path / f"results_{timestamp}.json")
    result_format = {k: str(v) for k, v in results.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)
    separator = lambda: print("=" * 60)
    separator()
    print("CLASSIFICATION RESULTS")
    separator()
    print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))


def _avg_key(ds: Dataset, key: str) -> float:
    return float(np.mean(ds[key]))


def show_statistics(features_dataset: Dataset) -> None:
    """Print similarity statistics for genuine and synthetic features.

    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with
        [label, sim_min, sim_max, idx_min, idx_max]."""

    genuine_dataset = features_dataset.filter(lambda x: x["label"] == 0, batched=False)
    synthetic_dataset = features_dataset.filter(lambda x: x["label"] == 1, batched=False)

    for key in ("sim_min", "sim_max", "idx_min", "idx_max"):
        genuine_avg_similarity = float(np.mean(genuine_dataset[key]))
        synthetic_avg_similarity = float(np.mean(synthetic_dataset[key]))
        print(f"""
        Average {key} (genuine): {genuine_avg_similarity:.4f}
        Average {key} (synthetic): {synthetic_avg_similarity:.4f}
        """)

    gen_sim = np.array(genuine_dataset["sim_max"])
    gen_idx = np.array(genuine_dataset["idx_max"])
    syn_sim = np.array(synthetic_dataset["sim_max"])
    syn_idx = np.array(synthetic_dataset["idx_max"])

    gen_diff = float(np.mean(gen_sim - gen_idx))
    syn_diff = float(np.mean(syn_sim - syn_idx))

    print(f"""
    Average (sim_max - idx_max) genuine: {gen_diff:.4f}
    Average (sim_max - idx_max) synthetic: {syn_diff:.4f}
    """)

    def overall_avg(ds: Dataset) -> float:
        idx = (_avg_key(ds, "idx_min") + _avg_key(ds, "idx_max")) / 2
        sim = (_avg_key(ds, "sim_min") + _avg_key(ds, "sim_max")) / 2
        return (idx + sim) / 2

    g_avg = overall_avg(genuine_dataset)
    s_avg = overall_avg(synthetic_dataset)

    print(f"""
    Overall average genuine cosine similarity: {g_avg:.4f}
    Overall average synthetic cosine similarity: {s_avg:.4f}
    Overall average cosine similarity: {(g_avg + s_avg) / 2:.4f}
    """)


def compare_decompositions(model_name, features_dataset: Dataset) -> None:
    """Plot wavelet sensitivity distributions.\n
    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with
        [label, sim_min, sim_max, idx_min, idx_max]."""

    data_frame = features_dataset.to_pandas()

    # Derive sensitivity as the average of min/max similarity bounds
    data_frame["sensitivity"] = (data_frame["sim_min"].values + data_frame["sim_max"].values) / 2  # type: ignore
    data_frame["index_data"] = (data_frame["idx_min"].values + data_frame["idx_max"].values) / 2  # type: ignore

    chart = (
        alt.Chart(data_frame)
        .mark_line(opacity=0.7)
        .encode(
            x=alt.X("sensitivity:Q", title="Cosine Similarity"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("label:N", legend=None),
        )
        .transform_density("sensitivity", as_=["sensitivity", "density"], groupby=["label"])
        .transform_filter((alt.datum.sensitivity <= 1.0) & (alt.datum.sensitivity >= -1.0))
        .properties(title=f"Sensitivity Distribution by Label\nModel: {model_name}", width=600, height=300)
    )
    chart_file = str(result_path / f"sensitivity_plot_{timestamp}.html")
    chart.save(chart_file)
    # chart.display()
