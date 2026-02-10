# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json
import time as timer_module
from pathlib import Path

import altair as alt
import numpy as np
from datasets import Dataset

from negate.train import TrainResult, get_time

timestamp = get_time()
result_path = Path(__file__).parent.parent / "results" / timestamp


def accuracy(train_result: TrainResult):
    """Print diagnostics and plots for a trained model.\n
    :param train_result: Result object from training."""
    try:
        from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
    except (ImportError, ModuleNotFoundError, Exception):
        raise RuntimeError("missing dependencies for xgboost. Please install using 'negate[xgb]'")

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


def _avg_key(ds: Dataset, key: str) -> float:
    arr = np.array(ds[key]).flatten()
    return float(np.mean(arr).item())


def show_statistics(features_dataset: Dataset, start_ns: int | float | None = None) -> None:
    """Print similarity statistics for genuine and synthetic features.

    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with
        [label, sim_min, sim_max, idx_min, idx_max]."""
    from negate import hyper_param, negate_d, negate_opt

    stats = {}

    genuine_dataset = features_dataset.filter(lambda x: x["label"] == 0, batched=False)
    synthetic_dataset = features_dataset.filter(lambda x: x["label"] == 1, batched=False)

    for key in ("sim_min", "sim_max", "idx_min", "idx_max"):
        genuine_values = np.array(genuine_dataset[key]).flatten()
        synthetic_values = np.array(synthetic_dataset[key]).flatten()
        genuine_avg_similarity = float(np.mean(genuine_values).item())
        synthetic_avg_similarity = float(np.mean(synthetic_values).item())
        stats[f"genuine_avg_{key}"] = genuine_avg_similarity
        stats[f"synthetic_avg_{key}"] = synthetic_avg_similarity
        print(f"""
        Average {key} (genuine): {genuine_avg_similarity:.4f}
        Average {key} (synthetic): {synthetic_avg_similarity:.4f}
        """)

    gen_sim = np.array(genuine_dataset["sim_max"]).flatten()
    gen_idx = np.array(genuine_dataset["idx_max"]).flatten()
    syn_sim = np.array(synthetic_dataset["sim_max"]).flatten()
    syn_idx = np.array(synthetic_dataset["idx_max"]).flatten()

    gen_diff = float(np.mean(gen_sim - gen_idx).item())
    syn_diff = float(np.mean(syn_sim - syn_idx).item())

    stats["genuine_avg_sim_max_minus_idx_max"] = gen_diff
    stats["synthetic_avg_sim_max_minus_idx_max"] = syn_diff

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

    stats["overall_avg_genuine_cosine_similarity"] = g_avg
    stats["overall_avg_synthetic_cosine_similarity"] = s_avg
    stats["overall_avg_cosine_similarity"] = (g_avg + s_avg) / 2

    print(f"""
    Overall average genuine cosine similarity: {g_avg:.4f}
    Overall average synthetic cosine similarity: {s_avg:.4f}
    Overall average cosine similarity: {(g_avg + s_avg) / 2:.4f}
    """)

    result_path.mkdir(parents=True, exist_ok=True)
    stats_file = str(result_path / f"stats_{timestamp}.json")

    if start_ns is not None:
        elapsed_ns = timer_module.perf_counter_ns() - start_ns
        stats["elapsed_ns"] = elapsed_ns
        stats["ns_as_human_time"] = f"{elapsed_ns / 1e9:.2}"

    stats_format = {k: str(v) for k, v in stats.items()}
    stats_format.update(negate_opt._asdict())
    stats_format.update(negate_d._asdict())
    stats_format.update(hyper_param._asdict())
    with open(stats_file, "tw", encoding="utf-8") as out_file:
        json.dump(stats_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)


def compare_decompositions(model_name, features_dataset: Dataset) -> None:
    """Plot wavelet sensitivity distributions.\n
    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with
        [label, sim_min, sim_max, idx_min, idx_max]."""

    data_frame = features_dataset.to_pandas()
    data_frame["sensitivity"] = (data_frame["sim_min"].values + data_frame["sim_max"].values) / 2  # type: ignore
    lower_bound = 0.800
    upper_bound = 0.900
    data_frame["is_within_range"] = (data_frame["sensitivity"] >= lower_bound) & (data_frame["sensitivity"] <= upper_bound)  # type: ignore

    chart = (
        alt.Chart(data_frame)
        .mark_line(opacity=0.7)
        .encode(
            x=alt.X("sensitivity:Q", title="Cosine Similarity"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("label:N"),
        )
        .transform_density("sensitivity", as_=["sensitivity", "density"], groupby=["label"])
        .transform_filter((alt.datum.sensitivity <= 1.0) & (alt.datum.sensitivity >= -1.0))
        .properties(title=f"Sensitivity by Label\nModel: {model_name}", width=600, height=300)
    )
    result_path.mkdir(parents=True, exist_ok=True)
    chart_file = str(result_path / f"sensitivity_plot_{timestamp}.html")
    chart.save(chart_file)

    synthetic_count = int(data_frame["is_within_range"].sum())  # type: ignore
    non_synthetic_count = len(data_frame) - synthetic_count  # type: ignore
    ratio = synthetic_count / non_synthetic_count if non_synthetic_count > 0 else float("inf")
    print(f"Synthetic to non-synthetic ratio: {ratio:.3f}")
