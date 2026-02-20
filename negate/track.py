# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pandas as pd
import numpy as np
from datasets import Dataset
from scipy.stats import iqr
from negate.plot import (
    graph_tail_separations,
    graph_wavelet,
    residual_keys,
    wavelet_keys,
    graph_residual,
    graph_kde,
    graph_cohen,
    vae_loss_keys,
    graph_vae_loss,
    result_path,
)


def compute_tail_separation(data_frame, residual_keys) -> pd.DataFrame:
    """Score metrics by class separation and long-tail outlier tendency.\n
    :param data_frame: Expanded DataFrame with residual columns.
    :param residual_keys: List of metric column names to analyze.
    :returns: DataFrame with scores sorted by separateness."""

    results = []
    for key in residual_keys:
        vals_0 = data_frame[data_frame["label"] == 0][key].dropna().values
        vals_1 = data_frame[data_frame["label"] == 1][key].dropna().values

        if len(vals_0) < 5 or len(vals_1) < 5:
            continue

        def tail_score(arr):
            q75, q25 = np.percentile(arr, [75, 25])
            iqr_ = q75 - q25
            if iqr_ == 0:
                return 0
            return (np.percentile(arr, 95) - np.median(arr)) / iqr_

        tail_0 = tail_score(vals_0)
        tail_1 = tail_score(vals_1)
        avg_tail = (tail_0 + tail_1) / 2

        p95_0, p95_1 = np.percentile(vals_0, 95), np.percentile(vals_1, 95)
        p05_0, p05_1 = np.percentile(vals_0, 5), np.percentile(vals_1, 5)

        overlap_gid = max(p05_0, p05_1) - min(p95_0, p95_1)
        pooled_iqr = (iqr(vals_0) + iqr(vals_1)) / 2

        separation_norm = abs((np.median(vals_0) - np.median(vals_1))) / (pooled_iqr if pooled_iqr > 0 else 1)
        combined_score = avg_tail * max(0, separation_norm)

        results.append({"metric": key, "tail_score": avg_tail, "separation": separation_norm, "combined_score": combined_score, "no_overlap": overlap_gid < 0})

    return pd.DataFrame(results).sort_values("combined_score", ascending=False)


def chart_decompositions(model_name: str, features_dataset: Dataset, vae_name) -> None:
    """Plot wavelet sensitivity distributions."""

    data_frame = features_dataset.to_pandas()
    expanded_frame = data_frame.explode("results").reset_index(drop=True)

    for key in wavelet_keys:
        expanded_frame[key] = expanded_frame["results"].apply(lambda x, k=key: float(np.mean(x[k])) if isinstance(x, dict) and k in x else None)

    for key in residual_keys:
        expanded_frame[key] = expanded_frame["results"].apply(lambda x, k=key: float(np.mean(x[k])) if isinstance(x, dict) and k in x else None)

    for key in vae_loss_keys:
        expanded_frame[key] = expanded_frame["results"].apply(lambda x, k=key: float(np.mean(x[k])) if isinstance(x, dict) and k in x else None)

    scores_dataframe = compute_tail_separation(expanded_frame, residual_keys)
    graph_tail_separations(model_name, scores_dataframe=scores_dataframe)
    graph_wavelet(model_name, wavelet_dataframe=expanded_frame)
    graph_residual(model_name, residual_dataframe=expanded_frame)
    graph_kde(model_name, residual_dataframe=expanded_frame)
    graph_cohen(model_name, residual_dataframe=expanded_frame)
    graph_vae_loss(model_name, vae_dataframe=expanded_frame)
    print(f"[TRACK] Saved plots to {result_path}")
