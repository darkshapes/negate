# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from negate import TrainResult, VAEModel, get_time, model_path


def in_console(train_result: TrainResult, vae_type: VAEModel) -> None:
    """Print diagnostics and plots for a trained model.\n
    :param train_result: Result object from training."""
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
    results_file = model_path("results.json")
    result_format = {k: str(v) for k, v in results.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)

    separator = lambda: print("=" * 60)
    separator()
    print("CLASSIFICATION RESULTS")
    separator()
    print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))


def on_graph(train_result: TrainResult) -> None:
    """Save and show PCA variance plots for a trained model.\n
    :param train_result: Result object from training."""

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray
    from sklearn.metrics import confusion_matrix

    X_train: NDArray = train_result.X_train
    X_train_pca = train_result.X_train_pca
    labels = train_result.labels
    y_plot = labels[: X_train.shape[0]]
    y_pred_proba = train_result.model.predict(train_result.d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    pca = train_result.pca
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color="aqua")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(range(min(20, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_[:20], color="aqua")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("First 20 Components")
    plt.tight_layout()
    plt.savefig(model_path("score_explained_variance.png"))
    plt.show()

    cm = confusion_matrix(train_result.y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.imshow(cm, interpolation="nearest", cmap="Reds")

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(["Real", "Synthetic"])
    ax.set_yticklabels(["Real", "Synthetic"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(cax)
    plt.savefig(model_path("score_confusion_matrix.png"))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Data (First Two Features)")
    plt.colorbar(label="Prediction")

    plt.subplot(1, 2, 2)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Transformed Data")
    plt.colorbar(label="Prediction")
    plt.tight_layout()
    plt.savefig(model_path("pca_transform_map.png"))
    plt.show()

    import seaborn as sns

    corr = np.corrcoef(X_train_pca, rowvar=False)
    upper_triangle_mask = np.triu(np.ones_like(corr, dtype=bool))

    # Get actual min/max from the lower triangle (excluding diagonal)
    lower_triangle = corr[np.tril_indices_from(corr, k=-1)]
    vmin = lower_triangle.min()
    vmax = lower_triangle.max()

    figure, ax = plt.subplots(figsize=(12, 10))
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
    )
    ax.set_title(f"Feature Correlation Heatmap (PCA Components)\nRange: [{vmin:.3e}, {vmax:.3e}]")
    plt.tight_layout()
    figure.savefig(model_path("correlation_heatmap.png"))
    plt.show()
