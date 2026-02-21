# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from PIL import Image
import pandas as pd
from pandas import Series
from negate.train import timestamp, result_path


plot_file = "plot_xp_data.json"

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


def save_frames(data_frame: pd.DataFrame, model_name: str) -> None:
    data_frame["model_name"] = model_name
    frames = data_frame.to_dict(orient="records")
    data_log = str(result_path / plot_file)
    result_path.mkdir(parents=True, exist_ok=True)
    with open(data_log, mode="tw+") as plot_data:
        json.dump(frames, plot_data, indent=4, ensure_ascii=False, sort_keys=False)


def load_frames(folder_path_name: str) -> tuple[pd.DataFrame, Series]:
    plot_path = Path(__file__).parent.parent / "results" / folder_path_name
    with open(str(plot_path / plot_file), "r") as plot_data:
        saved_frames = json.load(plot_data)
    xp_frames = pd.DataFrame.from_dict(json.loads(saved_frames))
    model_name = xp_frames.pop("model_name")
    return xp_frames, model_name


def invert_image(input_path: str, output_path: str) -> None:
    """Invert colors of a PNG image (create negative).\n
    :param input_path: Path to source PNG.
    :param output_path: Path for inverted output."""

    img = Image.open(input_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    r, g, b = img.split()
    r = Image.eval(r, lambda x: 255 - x)
    g = Image.eval(g, lambda x: 255 - x)
    b = Image.eval(b, lambda x: 255 - x)

    inverted = Image.merge("RGB", (r, g, b))
    inverted.save(output_path)


def graph_tail_separations(model_name: str, scores_dataframe: pd.DataFrame) -> None:
    """Plot tail-separation analysis for residual metrics.\n

    :param model_name: Name of the model being analyzed.
    :param data_frame: Expanded DataFrame with residual columns.
    :param residual_keys: List of metric column names."""

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(scores_dataframe) * 0.4)))

    colors = ["magenta" if no else "magenta" for no in scores_dataframe["no_overlap"].values]
    bars = axes[0].barh(range(len(scores_dataframe)), scores_dataframe["combined_score"], color=colors)

    axes[0].set_yticks(range(len(scores_dataframe)), labels=scores_dataframe["metric"])
    axes[0].set_xlabel("Tail x Separation Score")
    axes[0].set_title(f"Long-Tail Outlier Separability - {model_name}")
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
    axes[1].set_title(f"Metric Diagnostic - {model_name}")
    axes[1].grid(True, alpha=0.3)

    med_sep = scores_dataframe["separation"].median()
    med_tail = scores_dataframe["tail_score"].median()
    axes[1].axhline(med_sep, color="gray", linestyle="--", linewidth=0.5)
    axes[1].axvline(med_tail, color="gray", linestyle="--", linewidth=0.5)

    legend_elements_2 = [Patch(facecolor="green", label="No Overlap")]
    axes[1].legend(handles=legend_elements_2, loc="lower right")

    plt.tight_layout()
    tail_separation_plot = str(result_path / f"tail_separation_plot_{timestamp}.png")
    plt.savefig(tail_separation_plot)


def graph_wavelet(model_name: str, wavelet_dataframe: pd.DataFrame) -> None:
    """Plot wavelet sensitivity distributions.\n

    :param model_name: Name of the model being analyzed.
    :features_dataset: Dataset containing feature results.
    :param timecode: Timestamp for file naming."""

    fig, axes = plt.subplots(2, int(len(wavelet_keys) / 2), figsize=(12, 10))

    for idx, key in enumerate(wavelet_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "cyan"), (1, "red")]:
            subset = wavelet_dataframe[wavelet_dataframe["label"] == label_val][key].dropna()  # type: ignore dropna
            subset = pd.to_numeric(subset, errors="coerce").dropna()  # type: ignore dropna
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"{label_val} {'syn' if label_val == 1 else 'gnd'}", density=True, color=color)
        ax.set_title(f"{key} Distribution")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.suptitle(f"Wavelet Decomposition Comparison - {model_name}")
    sensitivity_plot = str(result_path / f"sensitivity_plot_{timestamp}.png")
    plt.savefig(sensitivity_plot)


def graph_residual(model_name: str, residual_dataframe: pd.DataFrame) -> None:
    num_keys = len(residual_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 17))

    for idx, key in enumerate(residual_keys):
        ax = axes.flat[idx - 1]
        data_by_label = []
        labels = []
        for label_val in [0, 1]:
            subset = residual_dataframe[residual_dataframe["label"] == label_val][key].dropna()  # type: ignore dropna
            if len(subset) > 0:
                data_by_label.append(subset.values)
                labels.append(f"{label_val} {'syn' if label_val == 1 else 'gnd'}")
        if data_by_label:
            ax.boxplot(data_by_label, labels=labels)
        ax.set_title(f"{key} by Label")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Residual Metrics Comparison - {model_name}")
    plt.tight_layout()

    residual_plot = str(result_path / f"residual_plot_{timestamp}.png")
    plt.savefig(residual_plot)


def graph_kde(model_name: str, residual_dataframe: pd.DataFrame) -> None:
    num_keys = len(residual_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 20))

    for idx, key in enumerate(residual_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "tab:cyan"), (1, "tab:red")]:
            subset = residual_dataframe[residual_dataframe["label"] == label_val][key].dropna()  # type: ignore dropna
            if len(subset) > 0:
                sns.kdeplot(data=subset, ax=ax, fill=True, alpha=0.4, label=f"L{label_val} {'syn' if label_val == 1 else 'gnd'}", color=color)  # type: ignore subset dataframe
        ax.set_title(key, fontsize=9)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # Only add legend if there are artists
            ax.legend()

    plt.suptitle(f"Residual Metrics KDE Comparison - {model_name}")
    plt.tight_layout()
    kde_plot = str(result_path / f"residual_kde_plot_{timestamp}.png")
    plt.savefig(kde_plot)


def graph_cohen(model_name: str, residual_dataframe: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, len(residual_keys) * 0.3)))

    effect_sizes = []
    for key in residual_keys:
        vals_0 = residual_dataframe[residual_dataframe["label"] == 0][key].dropna().values  # type: ignore dropna
        vals_1 = residual_dataframe[residual_dataframe["label"] == 1][key].dropna().values  # type: ignore dropna
        effect_size = 0

        if len(vals_0) > 1 and len(vals_1) > 1:
            mean_diff = abs(np.mean(vals_1) - np.mean(vals_0))  # type: ignore mean
            pooled_std = np.sqrt((np.std(vals_0, ddof=1) ** 2 + np.std(vals_1, ddof=1) ** 2) / 2)  # type: ignore std
            if pooled_std > 0 and not (np.isnan(mean_diff) or np.isnan(pooled_std)):
                effect_size = mean_diff / pooled_std

        effect_sizes.append(effect_size)

    heatmap_data = pd.DataFrame({"Effect Size": effect_sizes}, index=residual_keys)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="magma", ax=ax, cbar=False)

    plt.title(f"Class Separation (Cohen's d) - {model_name}", fontsize=11)
    effect_size_plot = str(result_path / f"effect_size_heatmap_{timestamp}.png")
    plt.savefig(effect_size_plot)


def graph_vae_loss(vae_name: str | list[str], vae_dataframe: pd.DataFrame) -> None:
    """Plot wavelet sensitivity distributions.\n

    :param model_name: Name of the model being analyzed.
    :features_dataset: Dataset containing feature results.
    :param timecode: Timestamp for file naming."""

    num_keys = len(vae_loss_keys)
    cols = int(round(num_keys**0.5))
    rows = (num_keys + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for idx, key in enumerate(vae_loss_keys):
        ax = axes.flat[idx]
        for label_val, color in [(0, "orange"), (1, "magenta")]:
            subset = vae_dataframe[vae_dataframe["label"] == label_val][key].dropna()  # type: ignore dropna
            subset = pd.to_numeric(subset, errors="coerce").dropna()  # type: ignore dropna
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"{label_val} {'syn' if label_val == 1 else 'gnd'}", density=True, color=color)
        ax.set_title(f"{key}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    plt.tight_layout()

    if isinstance(vae_name, list):
        vae_name = vae_name[0]
    plt.suptitle(f"VAE Loss Comparison - {vae_name[0]}")
    vae_plot = str(result_path / f"vae_plot{timestamp}.png")
    plt.savefig(vae_plot)
