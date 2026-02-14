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

    result_keys = list(expanded_frame["results"].iloc[0].keys())

    for key in result_keys:
        expanded_frame[key] = expanded_frame["results"].apply(lambda x: float(np.mean(x[key])) if not np.isinf(x[key]).all() else None)

    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(2, len(result_keys), figsize=(12, 10))

    for index, key in enumerate(result_keys):
        ax = axes.flat[index]
        for label_val, color in [(0, "cyan"), (1, "red")]:
            subset = expanded_frame[expanded_frame["label"] == label_val][key].dropna()
            subset = subset[~np.isinf(subset)]
            ax.hist(subset, bins=50, alpha=0.5, label=f"Label {label_val}", density=True, color=color)
        ax.set_title(f"{key} Distribution by Label")
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


# def graph_residuals(self, **kwargs):
#     fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10, 16))

#     ax[0, 0].imshow(numeric_image, cmap="gray")
#     ax[0, 0].set_title(f"Original Image {img_mean:.4f}")
#     ax[0, 1].imshow(np.log(img_mag), cmap="magma")
#     ax[0, 1].set_title("Original FFT Magnitude (log)")

#     ax[1, 0].imshow(self._normalize_to_uint8(diff_res), cmap="gray")
#     ax[1, 0].set_title(f"Difference of Gaussians {tc_dg_mean:.4f}")
#     ax[1, 1].imshow(np.log(ff_dg_mag), cmap="magma")
#     ax[1, 1].set_title("DoG FFT Magnitude (log)")

#     ax[2, 0].imshow(laplace_res, cmap="gray")
#     ax[2, 0].set_title(f"Laplacian Residual: {tc_lapl_mean:.4f}")
#     ax[2, 1].imshow(np.log(ff_lapl_mag), cmap="magma")
#     ax[2, 1].set_title("Laplacian FFT Magnitude (log)")

#     ax[3, 0].imshow(sobel_res, cmap="gray")
#     ax[3, 0].set_title(f"Sobel Residual: {tc_sobl_mean:.4f}")
#     ax[3, 1].imshow(np.log(ff_sobl_mag), cmap="magma")
#     ax[3, 1].set_title("Sobel FFT Magnitude (log)")

#     ax[4, 0].imshow(spectral_res, cmap="gray")
#     ax[4, 0].set_title(f"Spectral Residual: {tc_spc_mean:.4f}")
#     ax[4, 1].imshow(np.log(ff_spc_mag), cmap="magma")
#     ax[4, 1].set_title("Spectral FFT Magnitude (log)")
#     plt.tight_layout()
#     plt.savefig(str(Path(__file__).parent.parent / "results" / "gaussian" / name))
