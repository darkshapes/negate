# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from negate import TrainResult, VAEModel, get_time, generate_datestamp_path, model_path
import matplotlib.pyplot as plt


def in_console(train_result: TrainResult, vae_type: VAEModel) -> None:
    """Print diagnostics and plots for a trained model.\n
    :param train_result: Result object from training."""
    from pathlib import Path
    import shutil
    import json
    from pprint import pprint

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

    X_train = train_result.X_train
    pca = train_result.pca
    d_matrix_test = train_result.d_matrix_test
    model = train_result.model
    scale_pos_weight = train_result.scale_pos_weight
    X_train_pca = train_result.X_train_pca
    y_test = train_result.y_test
    labels = train_result.labels
    feature_matrix = train_result.feature_matrix
    seed = train_result.seed

    y_pred_proba = model.predict(d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    timestamp = get_time()

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
        "vae_type": vae_type.value,
    }

    pprint(results)
    results_file = generate_datestamp_path("results.json")
    result_format = {k: str(v) for k, v in results.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)
    shutil.copy(results_file, model_path / Path(results_file).name)  # type: ignore no overloads
    separator = lambda: print("=" * 60)
    separator()
    print("CLASSIFICATION RESULTS")
    separator()
    print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))


def on_graph(train_result: TrainResult) -> None:
    """Save and show PCA variance plots for a trained model.\n
    :param train_result: Result object from training."""
    import numpy as np
    from numpy.typing import NDArray
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    X_train: NDArray = train_result.X_train
    X_train_pca = train_result.X_train_pca
    labels = train_result.labels
    y_plot = labels[: X_train.shape[0]]
    y_pred_proba = train_result.model.predict(train_result.d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    pca = train_result.pca

    # Create a single figure with 6 subplots (2 rows × 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax_cum = axes[0, 0]
    ax_bar = axes[0, 1]
    ax_conf = axes[0, 2]
    ax_orig = axes[1, 0]
    ax_pca = axes[1, 1]
    ax_heat = axes[1, 2]

    # 1. Cumulative explained variance
    ax_cum.plot(np.cumsum(pca.explained_variance_ratio_), color="aqua")
    ax_cum.set_xlabel("Number of Components")
    ax_cum.set_ylabel("Cumulative Explained Variance")
    ax_cum.set_title("PCA Explained Variance")
    ax_cum.grid(True)

    # 2. First 20 components
    ax_bar.bar(
        range(min(20, len(pca.explained_variance_ratio_))),
        pca.explained_variance_ratio_[:20],
        color="aqua",
    )
    ax_bar.set_xlabel("Component")
    ax_bar.set_ylabel("Explained Variance Ratio")
    ax_bar.set_title("First 20 Components")

    # 3. Confusion matrix
    cm = confusion_matrix(train_result.y_test, y_pred)
    cax = ax_conf.imshow(cm, interpolation="nearest", cmap="Reds")
    ax_conf.set_xticks(np.arange(cm.shape[1]))
    ax_conf.set_yticks(np.arange(cm.shape[0]))
    ax_conf.set_xticklabels(["Real", "Synthetic"])
    ax_conf.set_yticklabels(["Real", "Synthetic"])
    plt.setp(ax_conf.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_conf.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("Actual")
    ax_conf.set_title("Confusion Matrix")
    fig.colorbar(cax, ax=ax_conf)

    # 4. Original data scatter
    ax_orig.scatter(X_train[:, 0], X_train[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
    ax_orig.set_xlabel("Feature 1")
    ax_orig.set_ylabel("Feature 2")
    ax_orig.set_title("Original Data (First Two Features)")
    # ax_orig.colorbar(label="Prediction")

    # 5. PCA transformed scatter
    ax_pca.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    ax_pca.set_title("PCA Transformed Data")
    # ax_pca.colorbar(label="Prediction")

    # 6. Correlation heatmap
    corr = np.corrcoef(X_train_pca, rowvar=False)
    upper_triangle_mask = np.triu(np.ones_like(corr, dtype=bool))
    lower_triangle = corr[np.tril_indices_from(corr, k=-1)]
    vmin = lower_triangle.min()
    vmax = lower_triangle.max()
    cmap = sns.diverging_palette(20, 230, as_cmap=True)
    sns.heatmap(
        corr,
        mask=upper_triangle_mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax_heat,
    )
    ax_heat.set_title(f"Feature Correlation Heatmap (PCA Components)\nRange: [{vmin:.3e}, {vmax:.3e}]")

    plt.tight_layout(pad=0.5)
    plt.savefig(generate_datestamp_path("combined_plots.png"))
