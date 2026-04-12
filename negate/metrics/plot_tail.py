# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Plot tail-separation analysis for residual metrics."""

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from negate.io.spec import Spec

wavelet_keys = ["min_warp", "max_warp", "min_base", "max_base"]
residual_keys = [
    "diff_mean",
    "diff_tc",
    "high_freq_ratio",
    "image_mean",
    "image_mean_ff",
    "image_std",
    "image_tc",
    "laplace_mean",
    "laplace_tc",
    "low_freq_energy",
    "max_fourier_magnitude",
    "max_magnitude",
    "mean_log_magnitude",
    "selected_patch_idx",
    "sobel_mean",
    "sobel_tc",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_tc",
]


def graph_tail_separations(spec: Spec, scores_dataframe) -> None:
    """Plot tail-separation analysis for residual metrics.

    :param spec: Configuration specification.
    :param scores_dataframe: DataFrame with residual columns.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(scores_dataframe) * 0.4)))

    colors = ["magenta" if no else "magenta" for no in scores_dataframe["no_overlap"].values]
    bars = axes[0].barh(range(len(scores_dataframe)), scores_dataframe["combined_score"], color=colors)

    axes[0].set_yticks(range(len(scores_dataframe)), labels=scores_dataframe["metric"])
    axes[0].set_xlabel("Tail x Separation Score")
    axes[0].set_title(f"Long-Tail Outlier Separability - {spec.model}")
    axes[0].axvline(x=0, color="magenta", linestyle="-", linewidth=0.5)

    for i, (score, no) in enumerate(zip(scores_dataframe["combined_score"], scores_dataframe["no_overlap"])):
        if no:
            axes[0].text(score + 0.02 * max(scores_dataframe["combined_score"]), i, "●", ha="left", va="center")

    legend_elements = [Patch(facecolor="silver", label="No Overlap (ideal)")]
    axes[0].legend(handles=legend_elements, loc="lower right")

    colors_scatter = ["magenta" if no else "tab:magenta" for no in scores_dataframe["no_overlap"].values]
    axes[1].scatter(scores_dataframe["tail_score"], scores_dataframe["separation"], c=colors_scatter, s=120)

    for _, row in scores_dataframe.iterrows():
        axes[1].text(row["tail_score"] + 0.02, row["separation"], row["metric"][:12], fontsize=8)

    axes[1].set_xlabel("Tail Score (heavy-tail tendency)")
    axes[1].set_ylabel("Separation (Cohen's d-like)")
    axes[1].set_title(f"Metric Diagnostic - {spec.model}")
    axes[1].grid(True, alpha=0.3)

    med_sep = scores_dataframe["separation"].median()
    med_tail = scores_dataframe["tail_score"].median()
    axes[1].axhline(med_sep, color="gray", linestyle="--", linewidth=0.5)
    axes[1].axvline(med_tail, color="gray", linestyle="--", linewidth=0.5)

    legend_elements_2 = [Patch(facecolor="green", label="No Overlap")]
    axes[1].legend(handles=legend_elements_2, loc="lower right")

    plt.tight_layout()
    tail_separation_plot = str(root_folder / "results" / f"tail_separation_plot_{timestamp}.png")
    plt.savefig(tail_separation_plot)
    plt.close()


def graph_wavelet(spec: Spec, wavelet_dataframe) -> None:
    """Plot wavelet sensitivity distributions.

    :param spec: Configuration specification.
    :param wavelet_dataframe: Dataset containing feature results.
    """

    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(2, int(len(wavelet_keys) / 2), figsize=(12, 10))

    for idx, key in enumerate(wavelet_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "cyan"), (1, "red")]:
            subset = wavelet_dataframe[wavelet_dataframe["label"] == label_val][key].dropna()
            subset = pd.to_numeric(subset, errors="coerce").dropna()
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"{label_val} {'syn' if label_val == 1 else 'gnd'}", density=True, color=color)
        ax.set_title(f"{key} Distribution")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.suptitle(f"Wavelet Decomposition Comparison - {spec.model}")
    sensitivity_plot = str(root_folder / "results" / f"sensitivity_plot_{timestamp}.png")
    plt.savefig(sensitivity_plot)
    plt.close()
