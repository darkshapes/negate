# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import numpy as np


def weight_syn_feat(entry: dict[str, np.ndarray | float | int]):
    """Use VAE BCE LOSS threshold, base wavelet mean and mean RRC fourier to determine Synthetic (SYN) Images
    :param entry: Image features and associated metadata used for classification.
    """
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


def weight_gne_feat(entry: dict[str, np.ndarray | float | int]):
    """Use Laplace/Sobel fourier mean, Max Fourier Magnitude and BCE loss to determine Genuine (GNE) Images
    :param entry: Image features and associated metadata used for classification.
    """
    # flux2 klein fp16, timm/vit-base dino v3 lvd
    #  dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5                  # strength of perturbation Default 0.5)

    laplace: np.ndarray = entry["laplace_mean"]  # type: ignore | Above ~4.2-4.3 for GNE
    sobel: np.ndarray = entry["sobel_mean"]  # type: ignore | Below 4 for GNE
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


def compute_combined_certainty(
    ae_result: dict[str, list[int] | np.ndarray],
    dc_result: dict[str, list[int] | np.ndarray],
) -> None:  # list[dict]:
    """
    Compute certainty scores by combining all available inference methods.

    Each method contributes a vote (0-1). Certainty is the sum normalized 0-5 scale.
    """
    combined = []

    # Ensure both have same length
    n_images = min(len(ae_result["gne"]), len(dc_result["gne"]))

    for i in range(n_images):
        # Extract votes (0 or 1)
        ae_gn_vote = ae_result["gne"][i]
        dc_gn_vote = dc_result["gne"][i]

        ae_syn_vote = ae_result["syn"][i]
        dc_syn_vote = dc_result["syn"][i]

        model_ae_vote = 1 - ae_result["unk"][i]  # Convert model output: 0->GNE, 1->SYN
        model_dc_vote = 1 - dc_result["unk"][i]

        # Aggregate GNE signals (inverse of SYN)
        gne_score = (
            ae_gn_vote  # AE heuristic says GNE
            + dc_gn_vote  # DC heuristic says GNE
            + model_ae_vote  # AE model says GNE
            + model_dc_vote  # DC model says GNE
        )

        # Aggregate SYN signals
        syn_score = (
            ae_syn_vote  # AE heuristic says SYN
            + dc_syn_vote  # DC heuristic says SYN
            + (1 - model_ae_vote)  # AE model says SYN
            + (1 - model_dc_vote)  # DC model says SYN
        )

        # Determine classification and confidence
        total_signals = 4  # 2 heuristics + 2 models

        if gne_score >= syn_score:
            label = "GNE"
            certainty = round((gne_score / total_signals) * 100, 1)
        else:
            label = "SYN"
            certainty = round((syn_score / total_signals) * 100, 1)

        combined.append(
            {
                "index": i,
                "label": label,
                "confidence_pct": certainty,
                "gne_votes": gne_score,
                "syn_votes": syn_score,
                "votes_detail": {"ae_heuristic": ae_gn_vote, "dc_heuristic": dc_gn_vote, "ae_model": model_ae_vote, "dc_model": model_dc_vote},
            }
        )
    separator = lambda: print("=" * 60)
    separator()
    print("CERTAINTY RESULTS")
    separator()
    for result in combined:
        print(f"IMG_{result['index']:02d}: {result['label']:3s} | Confidence: {result['confidence_pct']:5.1f}% | (GNE:{result['gne_votes']}, SYN:{result['syn_votes']})")
    # return combined


def compute_weighted_certainty(
    ae_result: dict[str, list[int] | np.ndarray],
    dc_result: dict[str, list[int] | np.ndarray],
    heuristic_weight: float = 1.5,
    model_weight: float = 1,  # Models less lessbased on your logs
) -> None:  # list[dict]:
    """
    Compute certainty scores by combining all available inference methods.\n
    Each method contributes a vote (0-1). Certainty is the sum normalized 0-5 scale.\n
    - Heuristics: ~85-100% accurate
    - Models: ~47-71% accurate
    """
    combined = []
    n_images = min(len(ae_result["gne"]), len(dc_result["gne"]))

    for i in range(n_images):
        gne_score = 0.0
        syn_score = 0.0

        # AE heuristic contribution
        if ae_result["gne"][i] == 1:
            gne_score += heuristic_weight
        else:
            syn_score += heuristic_weight

        # DC heuristic contribution
        if dc_result["gne"][i] == 1:
            gne_score += heuristic_weight
        else:
            syn_score += heuristic_weight

        # AE model contribution (unk: 0=GNE, 1=SYN)
        if ae_result["unk"][i] == 0:
            gne_score += model_weight
        else:
            syn_score += model_weight

        # DC model contribution
        if dc_result["unk"][i] == 0:
            gne_score += model_weight
        else:
            syn_score += model_weight

        max_possible = (heuristic_weight * 2) + (model_weight * 2)

        pred = "GNE" if gne_score > syn_score else "SYN"
        certainty = round((max(gne_score, syn_score) / max_possible) * 100, 1)

        combined.append({"index": i, "prediction": pred, "confidence_pct": certainty, "score_diff": abs(gne_score - syn_score)})
        # Print formatted results
        separator = lambda: print("=" * 60)
        separator()
        print("CERTAINTY RESULTS")
        separator()
        for result in combined:
            print(f"IMG_{result['index']:02d}: {result['prediction']:3s} | Confidence: {result['confidence_pct']:5.1f}% | {result['score_diff']})")
    # return combined
