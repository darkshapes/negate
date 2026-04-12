# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Plot residual metrics analysis for AI detection."""

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from negate.io.spec import Spec
import numpy as np
import pandas as pd


def graph_residual(spec: Spec, residual_dataframe) -> None:
    """Plot boxplots for residual metrics by label.

    :param spec: Configuration specification.
    :param residual_dataframe: Dataset containing feature results.
    """

    num_keys = len(residual_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 17))

    for idx, key in enumerate(residual_keys):
        ax = axes.flat[idx - 1]
        data_by_label = []
        labels = []
        for label_val in [0, 1]:
            subset = residual_dataframe[residual_dataframe["label"] == label_val][key].dropna()
            if len(subset) > 0:
                data_by_label.append(subset.values)
                labels.append(f"{label_val} {'syn' if label_val == 1 else 'gnd'}")
        if data_by_label:
            ax.boxplot(data_by_label, labels=labels)
        ax.set_title(f"{key} by Label")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Residual Metrics Comparison - {spec.model}")
    plt.tight_layout()

    residual_plot = str(root_folder / "results" / f"residual_plot_{timestamp}.png")
    plt.savefig(residual_plot)
    plt.close()


def graph_kde(spec: Spec, residual_dataframe) -> None:
    """Plot KDE distributions for residual metrics by label.

    :param spec: Configuration specification.
    :param residual_dataframe: Dataset containing feature results.
    """

    num_keys = len(residual_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 20))

    for idx, key in enumerate(residual_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "tab:cyan"), (1, "tab:red")]:
            subset = residual_dataframe[residual_dataframe["label"] == label_val][key].dropna()
            if len(subset) > 0:
                sns.kdeplot(data=subset, ax=ax, fill=True, alpha=0.4, label=f"L{label_val} {'syn' if label_val == 1 else 'gnd'}", color=color)
        ax.set_title(key, fontsize=9)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    plt.suptitle(f"Residual Metrics KDE Comparison - {spec.model}")
    plt.tight_layout()
    kde_plot = str(root_folder / "results" / f"residual_kde_plot_{timestamp}.png")
    plt.savefig(kde_plot)
    plt.close()


def graph_cohen(spec: Spec, residual_dataframe) -> None:
    """Plot Cohen's d effect size heatmap for class separation.

    :param spec: Configuration specification.
    :param residual_dataframe: Dataset containing feature results.
    """

    fig, ax = plt.subplots(figsize=(10, max(4, len(residual_keys) * 0.3)))

    effect_sizes = []
    for key in residual_keys:
        vals_0 = residual_dataframe[residual_dataframe["label"] == 0][key].dropna().values
        vals_1 = residual_dataframe[residual_dataframe["label"] == 1][key].dropna().values
        effect_size = 0

        if len(vals_0) > 1 and len(vals_1) > 1:
            mean_diff = abs(np.mean(vals_1) - np.mean(vals_0))
            pooled_std = np.sqrt((np.std(vals_0, ddof=1) ** 2 + np.std(vals_1, ddof=1) ** 2) / 2)
            if pooled_std > 0 and not (np.isnan(mean_diff) or np.isnan(pooled_std)):
                effect_size = mean_diff / pooled_std

        effect_sizes.append(effect_size)

    heatmap_data = pd.DataFrame({"Effect Size": effect_sizes}, index=residual_keys)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="magma", ax=ax, cbar=False)

    plt.title(f"Class Separation (Cohen's d) - {spec.model}", fontsize=11)
    effect_size_plot = str(root_folder / "results" / f"effect_size_heatmap_{timestamp}.png")
    plt.savefig(effect_size_plot)
    plt.close()
