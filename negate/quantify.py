# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from numpy import real
import re, pandas as pd


def graph_result(residuals) -> None:
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.histplot(residuals["fractal_complexity"], label="Fractal Complexity", color="aqua", kde=True)
    plt.legend()
    plt.show()

    sns.histplot(residuals["texture_complexity"], label="Texture Complexity", color="aqua", kde=True)
    plt.legend()
    plt.show()


def graph_comparison(synthetic_stats, human_origin_stats) -> None:
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.histplot(synthetic_stats[0], label="Synthetic Fractal Complexity", color="red", kde=True)
    sns.histplot(human_origin_stats[0], label="Human Origin Fractal Complexity", color="aqua", kde=True)
    plt.legend()
    plt.show()

    sns.histplot(synthetic_stats[1], label="Synthetic Texture Complexity", color="red", kde=True)
    sns.histplot(human_origin_stats[1], label="Human Origin Texture Complexity", color="aqua", kde=True)
    plt.legend()
    plt.show()

    print(f"synthetic {[i.describe() for i in synthetic_stats]}")
    print(f"human origin {[i.describe() for i in human_origin_stats]}")
    print(f"synthetic {[i.quantile(0.25) for i in synthetic_stats]}")
    print(f"synthetic {[i.quantile(0.75) for i in synthetic_stats]}")
    print(f"human origin {[i.quantile(0.25) for i in synthetic_stats]}")
    print(f"human origin {[i.quantile(0.75) for i in synthetic_stats]}")
    lower, upper = -2.20, -1.80  # choose from visual inspection
    human_origin_outliers = [(human_origin_stats[0] < lower) | (human_origin_stats[0] > upper), (human_origin_stats[0] < lower)]
    synthetic_inliers = [(synthetic_stats[0] >= lower) & (synthetic_stats[0] <= upper), (synthetic_stats[0] >= lower)]
    print(f"human_origin boolean : {human_origin_outliers}")
    print(f"synthetic boolean: {synthetic_inliers}")


def flag_synthetic_or_human_origin(residuals: list) -> None:
    fractal = pd.DataFrame(residuals["fractal_complexity"])
    texture = pd.DataFrame(residuals["texture_complexity"])
    residuals = pd.concat([fractal, texture], axis=1)

    X_new = extract_features(Path("assets/real/img_001.png"))
    pred = model.predict(X_new)
    label_map = {0: "synthetic", 1: "real"}
    print(f"Predicted class: {label_map[pred[0]]}")
    proba = model.predict_proba(X_new)[0]
    print(f"Confidence: synthetic={proba[0]:.2f}, real={proba[1]:.2f}")
    # return pd.concat([fractal, texture], axis=1)
    # lower, upper = -2.20, -1.80  # choose from visual inspection
    # mu_syn = residuals["fractal_complexity"].mean()
    # sigma_syn = residuals["fractal_complexity"].std()
    # real_filtered = abs(residuals["fractal_complexity"] - mu_syn) > 2 * sigma_syn
    # print(f"filtered syn {real_filtered}")
    # # print(f"is synthetic? : {[(residuals['fractal_complexity'] < lower) | (residuals['fractal_complexity'] > upper)]}")
    # # Q1 = residuals["fractal_complexity"].quantile(0.25)
    # # Q3 = residuals["fractal_complexity"].quantile(0.75)
    # # IQR = Q3 - Q1
    # # lower_bound = Q1 - 1.5 * IQR
    # # upper_bound = Q3 + 1.5 * IQR


# from math import isclose
#     from decimal import Decimal

#     texture_dimension = tc.mean()
#     lower, upper = -2.20, -1.80  # choose from visual inspection
#     real_outliers = real[(real["fractal_complexity"] < lower) | (real["fractal_complexity"] > upper)]
#     synthetic_inliers = synthetic[(synthetic["fractal_complexity"] >= lower) & (synthetic["fractal_complexity"] <= upper)]
#     fractal = isclose(fractal_dimension, Decimal("-1.98"), rel_tol=60e-3)
#     texture = isclose(texture_dimension, Decimal("-7.2"), rel_tol=35e-3)

#     fractal_min = abs(fractal_dimension * 100) < 197
#     fractal_max = abs(fractal_dimension * 100) > 207

#     texture_min = abs(texture_dimension * 100) > 675
#     texture_max = abs(texture_dimension * 100) < 880
#     self.console(((fractal, texture), (abs(fractal_dimension * 100), fractal_max, fractal_min, abs(texture_dimension * 100), texture_min, texture_max)))
#     # self.verdict =
#     # self.console(("verdict" == "passed")))
#     # else:
#     # self.verdict =
#     # self.console(("verdict" == "failed")))
