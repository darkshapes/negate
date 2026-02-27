# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Genuine
DC GNE Probability (GNE_HEUR): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Heuristic Accuracy: 100.00%
AE GNE Probability (GNE_HEUR): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0] Heuristic Accuracy: 85.71%
AE Decision Tree Model result: [0 0 0 1 0 0 0 0 0 0 0 1 1 1] Model Accuracy: 71.43%
DC Decision Tree Model result: [0 0 0 0 0 0 0 1 0 0 0 1 1 1] Model Accuracy: 71.43%

Synthetic
AE SYN Probability (SYN HEUR): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] Heuristic Accuracy: 100.00%
DC SYN Probability (SYN HEUR): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] Heuristic Accuracy: 100.00%
AE Decision Tree Model result: [1 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 0 0] Model Accuracy: 52.63%
DC Decision Tree Model result: [1 0 0 0 1 0 0 0 0 1 1 1 1 1 1 0 0 1 0] Model Accuracy: 47.37%

"""

import numpy as np


# def weight_syn_feat(entry: dict[str, np.ndarray | float | int]):
#     """Use VAE BCE LOSS threshold, base wavelet mean and mean RRC fourier to determine Synthetic (SYN) Images
#     :param entry: Image features and associated metadata used for classification.
#     """
#     thresh = 0.5
#     bce_threshold = -80  # SYN has very negative bce_loss (< -80 typically)
#     ff_threshold = 0.0  # SYN tends to be higher in mean ff magnitude
#     min_base_threshold = 1000

#     bce_loss = entry["bce_loss"]  # bce_loss is much more negative in SYN (around -100+ vs GNE around -50)
#     image_mean_ff = entry["image_mean_ff"]  # Another indicator: Mean is dominated by negatives for SYN
#     min_base = entry["min_base"]  # min_base heuristic: GNE clusters around 1000-1300 range more tightly
#     score = 1

#     if bce_loss is not None:
#         if bce_loss < -150 or (-15 < bce_loss < -5):
#             score += 1  # More likely SYN
#         if bce_loss > -20 or bce_loss < -170:
#             score += 1
#         # if bce_threshold < bce_loss < -50 and min_base_threshold < min_base < 1400:
#         #     score -= 1

#     if image_mean_ff > ff_threshold:
#         score += 1
#     # elif bce_loss is not None and -100 > bce_loss > -50:
#     #     score -= 1  # More likely GNE

#     if min_base is not None:
#         # if min_base_threshold <= min_base <= 1350:
#         #     score -= 1  # More likely GNE (tighter cluster)
#         if min_base > 4000 or min_base < 200:
#             score += 1

#     confidence = round((score / 5) * 100)
#     probability = float(score / 5) if score != 0 else 0
#     type_name = "SYN" if probability > thresh else "GNE"
#     return type_name, confidence


# def weight_gne_feat(entry: dict[str, np.ndarray | float | int]):
#     """Use Laplace/Sobel fourier mean, Max Fourier Magnitude and BCE loss to determine Genuine (GNE) Images
#     :param entry: Image features and associated metadata used for classification.
#     """
#     thresh = 0.5
#     # flux2 klein fp16, timm/vit-base dino v3 lvd
#     #  dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5                  # strength of perturbation Default 0.5)

#     laplace: np.ndarray = entry["laplace_mean"]  # type: ignore | Above ~4.2-4.3 for GNE
#     sobel: np.ndarray = entry["sobel_mean"]  # type: ignore | Below 4 for GNE
#     max_ff_mag = entry["max_fourier_magnitude"]  # Above 1500 for GNE
#     bce_loss = entry["bce_loss"]  # bce_los GNE around -50 -60 is more likely to be GNE)

#     score = 1
#     if laplace is not None and any(x > 4.2 for x in laplace):
#         score += 1

#     if sobel is not None and any(x < 4 for x in sobel):
#         score += 1

#     if max_ff_mag is not None and max_ff_mag > 1500:
#         score += 1

#     if -50 > bce_loss > -60:
#         score += 1

#     confidence = round((score / 5) * 100)
#     probability = float(score / 5) if score != 0 else 0
#     type_name = "GNE" if probability > thresh else "SYN"
#     return type_name, confidence


def weight_syn_feat(entry):  # weight bce
    bce_threshold = -80  # SYN has very negative bce_loss (< -80 typically)
    ff_threshold = 0.0  # SYN tends to be higher in mean ff magnitude
    min_base_threshold = 100

    bce_loss = entry["bce_loss"]  # bce_loss is much more negative in SYN (around -100+ vs GNE around -50)
    image_mean_ff = entry["image_mean_ff"]  # Another indicator: Mean is dominated by negatives for SYN
    min_base = entry["min_base"]  # min_base heuristic: GNE clusters around 1000-1300 range more tightly
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
    return score, confidence


def weight_gne_feat(entry):
    # flux2 klein fp16, timm/vit-base dino v3 lvd
    #  dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5                  # strength of perturbation Default 0.5)

    laplace = entry["laplace_mean"]  # Above ~4.2-4.3 for GNE
    sobel = entry["sobel_mean"]  # Below 4 for GNE
    max_ff_mag = entry["max_fourier_magnitude"]  # Above 1500 for GNE
    bce_loss = entry["bce_loss"]  # bce_los GNE around -50 -60 is more likely to be GNE)

    score = 2
    confidence = 0
    if laplace is not None and any(x > 4.2 for x in laplace):
        confidence += 1
        score -= 1

    if sobel is not None and any(x < 4 for x in sobel):
        confidence += 1
        score -= 1

    if max_ff_mag is not None and max_ff_mag > 1500:
        confidence += 1
        score -= 1

    if -50 > bce_loss > -60:
        confidence += 1
        score += 0.6

    return score, confidence


def heuristic_accuracy(result, dc=True):
    """Calculate accuracy for heuristic evaluation runs"""
    gt_labels = (
        [
            1,
        ]
        * len(result)
        if dc
        else [
            0,
        ]
        * len(result)
    )
    acc = float(np.mean([p == g for p, g in zip(result, gt_labels)]))
    print(f"Heuristic Accuracy: {acc:.2%}")
    return result


def model_accuracy(result: np.ndarray, label: int | None = None, thresh: float = 0.5) -> list[tuple[str, int]]:
    """Convert probability array to tuple format (label, confidence)."""

    thresh = 0.5
    model_pred = (result > thresh).astype(int)
    if label is not None:
        ground_truth = np.full(model_pred.shape, label, dtype=int)
        acc = float(np.mean(model_pred == ground_truth))
        print(f"Model Accuracy: {acc:.2%}")

    type_conf = []
    for x in result:
        prob = float(x)
        conf = round((1 - prob) * 100) if prob < thresh else round(prob * 100)
        label_out = "GNE" if prob < thresh else "SYN"
        type_conf.append((label_out, conf))
    return type_conf


def compute_weighted_certainty(
    ae_inference: dict[str, list[tuple[str, int]]],
    dc_inference: dict[str, list[tuple[str, int]]],
    heur_weight: float = 1.0,
    model_weight: float = 0.5,
) -> list[tuple[str, int]]:
    """
    Compute certainty scores by combining all available inference methods.\n
    Each method contributes a vote (unk: 0=GNE, 1=SYN)). Certainty is the sum normalized 0-5 scale.\n

    """

    n_images = min(len(ae_inference["gne"]), len(dc_inference["gne"]))
    inferences = []
    print(f"ae_inference: {ae_inference['syn']}")
    print(f"dc_inference: {dc_inference['gne']}")
    for i in range(n_images):
        syn_weight: tuple[str, int] = ae_inference["syn"][i]
        gne_weight: tuple[str, int] = dc_inference["gne"][i]
        unk_weight: tuple[str, int] = ae_inference["unk"][i]
        candidates = [
            (syn_weight, syn_weight[1] * heur_weight),
            (gne_weight, gne_weight[1] * heur_weight),
            (unk_weight, unk_weight[1] * model_weight),
        ]
        estimation, _ = max(candidates, key=lambda x: x[1])
        print(estimation)
        inferences.append((estimation))
    return inferences
