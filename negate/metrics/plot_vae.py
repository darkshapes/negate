# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Plot VAE loss and training variance."""

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

from negate.io.spec import Spec, TrainResult
from numpy.typing import NDArray

vae_loss_keys = [
    "l1_loss",
    "mse_loss",
    "perturbed_l1_loss",
    "perturbed_mse_loss",
    "kl_loss",
    "bce_loss",
    "perturbed_kl_loss",
    "perturbed_bce_loss",
]


def graph_vae_loss(spec: Spec, vae_dataframe) -> None:
    """Plot VAE loss component distributions.

    :param spec: Configuration specification.
    :param vae_dataframe: Dataset containing VAE loss results.
    """

    import numpy as np
    import pandas as pd
    import seaborn as sns

    num_keys = len(vae_loss_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for idx, key in enumerate(vae_loss_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "orange"), (1, "magenta")]:
            subset = vae_dataframe[vae_dataframe["label"] == label_val][key].dropna()
            subset = pd.to_numeric(subset, errors="coerce").dropna()
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"{label_val} {'syn' if label_val == 1 else 'gnd'}", density=True, color=color)
        ax.set_title(f"{key}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()

    vae_name = spec.vae[0] if isinstance(spec.vae, list) else spec.vae
    plt.suptitle(f"VAE Loss Comparison - {vae_name}")
    vae_plot = str(root_folder / "results" / f"vae_plot{timestamp}.png")
    plt.savefig(vae_plot)
    plt.close()


def graph_train_variance(train_result: TrainResult, spec: Spec) -> None:
    """Save and show PCA variance plots for a trained model.

    :param train_result: Result object from training.
    :param spec: Configuration specification.
    """

    import numpy as np

    X_train: NDArray = train_result.X_train
    X_train_pca = train_result.X_train_pca
    labels = train_result.labels
    y_plot = labels[: X_train.shape[0]]
    y_pred_proba = train_result.model.predict(train_result.d_matrix_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    pca = train_result.pca

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax_cum = axes[0, 0]
    ax_bar = axes[0, 1]
    ax_conf = axes[0, 2]
    ax_orig = axes[1, 0]
    ax_pca = axes[1, 1]

    ax_cum.plot(np.cumsum(pca.explained_variance_ratio_), color="aqua")
    ax_cum.set_xlabel("Number of Components")
    ax_cum.set_ylabel("Cumulative Explained Variance")
    ax_cum.set_title("PCA Explained Variance")
    ax_cum.grid(True)

    ax_bar.bar(range(min(20, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_[:20], color="aqua")
    ax_bar.set_xlabel("Component")
    ax_bar.set_ylabel("Explained Variance Ratio")
    ax_bar.set_title("First 20 Components")

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

    ax_orig.scatter(X_train[:, 0], X_train[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
    ax_orig.set_xlabel("Feature 1")
    ax_orig.set_ylabel("Feature 2")
    ax_orig.set_title("Original Data (First Two Features)")

    if X_train_pca.shape[1] < 2:
        ax_pca.text(0.5, 0.5, f"Insufficient PCA components\n(only {X_train_pca.shape[1]} found)", ha="center", va="center", transform=ax_pca.transAxes)
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("(none)")
    else:
        ax_pca.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_plot, cmap="coolwarm", edgecolor="k")
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")

    ax_pca.set_title("PCA Transformed Data")

    plt.tight_layout(pad=0.5)
    combined_name = spec.vae[0] if isinstance(spec.vae, list) else spec.vae
    plt.suptitle(f"Training Variance - {combined_name} {spec.model}")
    combined_plots = str(root_folder / "results" / f"combined_plots{timestamp}.png")
    plt.savefig(combined_plots)
    plt.close()
