# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from datetime import datetime
from pathlib import Path

import altair as alt


def compare_decompositions(features_dataset) -> None:
    """Plot wavelet similarity distributions.\n
    :param features_dataset: Dataset from WaveletAnalyzer.decompose() with [label, sim_min, sim_max, idx_min, idx_max].\n
    :return: None."""

    df = features_dataset.to_pandas()

    # Derive sensitivity as the average of min/max similarity bounds
    df["sensitivity"] = (df["sim_min"] + df["sim_max"]) / 2

    chart = (
        alt.Chart(df)
        .mark_line(opacity=0.7)
        .encode(
            x=alt.X("sensitivity:Q", title="Cosine Similarity"),
            y=alt.Y("density:Q", title="Density"),
            color="label:N",
        )
        .transform_density("sensitivity", as_=["sensitivity", "density"], groupby=["label"])
        .transform_filter((alt.datum.sensitivity <= 1.0) & (alt.datum.sensitivity >= -1.0))
        .properties(title="Sensitivity Distribution by Label", width=600, height=300)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = Path(__file__).parent.parent / "results"
    chart.save(str(result_path / f"sensitivity_plot_{timestamp}.html"))
    chart.display()
