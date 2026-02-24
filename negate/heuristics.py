# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import json


def classify_gnf_or_syn(data_path: str) -> dict[str, dict[str, int | str]]:
    """
    Returns 0 for GNE-like results, 1 for SYN-like results determined by simple heuristic based on observed patterns:\n
    :param data_path: Path to json file with saved parameter data"""
    bce_threshold = -80  # SYN has very negative bce_loss (< -80 typically)
    ff_threshold = 0.0  # SYN tends to be higher in mean ff magnitude
    min_base_threshold = 100

    result = {}
    with open(data_path, "r") as json_file:
        json_data = json.load(json_file)
    for index, entry in enumerate(json_data):
        bce_loss = entry["results"][0]["bce_loss"]  # bce_loss is much more negative in SYN (around -100+ vs GNE around -50)
        image_mean_ff = entry["results"][0]["image_mean_ff"]  # Another indicator: Mean is dominated by negatives for SYN
        min_base = entry["results"][0]["min_base"]  # min_base heuristic: GNE clusters around 1000-1300 range more tightly
        score = 0
        confidence = 1

        if bce_loss is not None:
            if bce_loss < -150 or (-15 < bce_loss < -5):
                score += 1  # More likely SYN
            if bce_loss > -20 or bce_loss < -170:
                score += 1
            if bce_threshold < bce_loss < -50 and min_base_threshold < min_base < 1400:
                score -= 1
            confidence += 1

        if image_mean_ff > ff_threshold:
            score += 1
        elif bce_loss is not None and -100 > bce_loss > -50:
            score -= 1  # More likely GNE

        if min_base is not None:
            if min_base_threshold <= min_base <= 1350:
                score -= 1  # More likely GNE (tighter cluster)
            if min_base > 4000 or min_base < 200:
                score += 1
            confidence += 1

        result[str(index)] = {"score": score, "class": "SYN" if score > 0 else "GNE", "confidence": confidence}
    return result
