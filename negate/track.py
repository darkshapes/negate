# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from matplotlib import pyplot as plt

from negate.train import get_time

timestamp = get_time()
result_path = Path(__file__).parent.parent / "results" / timestamp
plot_file = "plot_xp_data.json"


def graph_decompositions(folder_name: str) -> None:
    plot_path = Path(__file__).parent.parent / "results" / folder_name
    with open(str(plot_path / plot_file), "r") as plot_data:
        saved_frames = json.load(plot_data)

    xp_frames = pd.DataFrame()
    xp_frames.from_dict(saved_frames)
    model_name = xp_frames.pop("model_name")

    wavelet_keys = ["min_warp", "max_warp", "min_base", "max_base"]
    residual_keys = [
        "all_magnitudes",
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for idx, key in enumerate(wavelet_keys):
        ax = axes.flat[idx] if len(wavelet_keys) > 1 else axes
        for lbl, col in [(0, "cyan"), (1, "red")]:
            vals = xp_frames.loc[xp_frames["label"] == lbl, key].dropna()
            ax.hist(vals.astype(float), bins=50, alpha=0.5, label=f"Label {lbl}", density=True, color=col)
        ax.set_title(f"{key} Distribution by Label")
        ax.legend()
    plt.suptitle(f"Wavelet Decomposition Comparison - {model_name}")
    plt.tight_layout()

    wav_log = str(plot_path / f"sensitivity_plot_{timestamp}.png")
    plt.savefig(wav_log)

    fig, axes = plt.subplots(5, 4, figsize=(14, 8))
    for idx, key in enumerate(residual_keys):
        ax = axes.flat[idx]

        series = (
            xp_frames[key]  # type: ignore arg is not missing for columns
            .apply(lambda v: (v.tolist() if isinstance(v, np.ndarray) else v) if isinstance(v, (list, np.ndarray)) else [v] if v is not None else [])
            .explode()
            .apply(lambda x: float(x) if not isinstance(x, (list, np.ndarray)) else np.nan)
            .astype(float)
        )
        data_by_label = [series[xp_frames["label"] == lbl].dropna() for lbl in [0, 1]]  # type: ignore dropna
        labels = [f"Label {lbl}" for lbl in [0, 1]]

        ax.boxplot(data_by_label, labels=labels, notch=True)
        ax.set_title(key)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Residual Metrics Comparison - {model_name}")
    plt.tight_layout()

    res_log = str(plot_path / f"residual_plot_{timestamp}.png")
    plt.savefig(res_log)


def compare_decompositions(model_name: str, features_dataset: Dataset) -> None:
    """Plot wavelet and residual metric distributions for both label values.\n
    :param model_name : Name of the model whose results are visualised.
    :param features_dataset : HuggingFace ``Dataset`` containing a ``label`` column and a nested ``results`` dictionary for each example.
    """

    wavelet_keys = ["min_warp", "max_warp", "min_base", "max_base"]
    residual_keys = [
        "all_magnitudes",
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

    data_frame = features_dataset.to_pandas()
    xp_frames = data_frame.copy()  # type: ignore can access copy stfu

    xp_frames["flat_res"] = xp_frames["results"].apply(lambda x: x.get("0", {}) if isinstance(x, dict) else {})  # Extract the inner dict (under key “0”) once

    for key in wavelet_keys:
        xp_frames[key] = xp_frames["flat_res"].apply(lambda d, k=key: float(d[k]) if k in d else None)

    for key in residual_keys:
        xp_frames[key] = xp_frames["flat_res"].apply(lambda d, k=key: d.get(k))

    xp_frames["model_name"] = model_name
    result_path.mkdir(parents=True, exist_ok=True)
    config_name = "config.toml"
    shutil.copy(str(Path(__file__).parent.parent / "config" / config_name), str(result_path / config_name))

    xp_frames = xp_frames.to_json()
    data_log = str(result_path / plot_file)
    with open(data_log, mode="tw+") as plot_data:
        json.dump(xp_frames, plot_data, indent=4, ensure_ascii=False, sort_keys=False)
