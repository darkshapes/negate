# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

import numpy as np
from datasets import Dataset

from negate.train import get_time

timestamp = get_time()
result_path = Path(__file__).parent.parent / "results" / timestamp


def compare_decompositions(model_name, features_dataset: Dataset) -> None:
    """Plot wavelet sensitivity distributions.
    :param model_name: Name of the model.
    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with
        [label, min_warp, max_warp, min_base, max_base].
    """
    import matplotlib.pyplot as plt

    data_frame = features_dataset.to_pandas()
    expanded_frame = data_frame.explode("results").reset_index(drop=True)

    # Convert result strings to list if needed
    if isinstance(expanded_frame["results"].iloc[0], str):
        import ast

        expanded_frame["results"] = expanded_frame["results"].apply(ast.literal_eval)

    # Extract mean values from each dict key's numpy array
    for key in ["min_warp", "max_warp", "min_base", "max_base"]:
        expanded_frame[key] = expanded_frame["results"].apply(lambda x: float(np.mean(x[key])) if not np.isinf(x[key]).all() else None)

    # Plot all four fields with smooth lines
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, metric in enumerate(["min_warp", "max_warp", "min_base", "max_base"]):
        ax = axes.flat[i]
        for label_val, color in [(0, "cyan"), (1, "red")]:
            subset = expanded_frame[expanded_frame["label"] == label_val][metric].dropna()
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"Label {label_val}", density=True, color=color)
        ax.set_title(f"{metric} Distribution by Label")
        ax.legend()

    plt.suptitle(f"Wavelet Decomposition Comparison - {model_name}")
    plt.title(f"Wavelet Decomposition Comparison - {model_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Warp/Base Values")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and show
    result_path.mkdir(parents=True, exist_ok=True)
    log = str(result_path / f"sensitivity_plot_{timestamp}.png")
    plt.savefig(log)
    # plt.show()
