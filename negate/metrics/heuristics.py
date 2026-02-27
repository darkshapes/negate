# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from pprint import pprint

from typing import Any
import numpy as np


# Heuristics are not in use, but left for reference example
# These ended up not working as effectively as the decision tree


def weight_dc_gne(entry: dict[str, Any]) -> float:
    """Use DC metrics for the classification of GNE origin
    :param entry: Image features and associated metadata used for classification.

    only applies to 20260225_221149:
    dc-ae-f32c32-sana-1.1-diffusers, timm/vit-base dino v3 lvd
    dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5
    """
    max_warp: float = entry["max_warp"]  # max_warp larger than 0.4 but less than 0.6 predominantly GNE
    diff_tc: list = entry["diff_tc"]  # type: ignore diff_tc less than 20 largely GNE
    max_base = entry["max_base"]  # higher than 10300 is GNE
    laplace_mean: list = entry["laplace_mean"]  # more than 4.3 strong indicator of GNE

    total = [0.66, 0.66, 0.66, 0.66]
    score = []

    if 0.4 < max_warp < 0.6:
        score.append(0)

    if any(x < 20 for x in diff_tc):
        score.append(1)

    if max_base > 10300:
        score.append(2)

    if any(x > 7 for x in laplace_mean):
        score.append(3)

    probability = np.sum([total[x] for x in score])
    return probability if probability.item() > 0 else 0


def weight_ae_gne(entry: dict[str, Any]) -> float:
    """Use AE metrics for the classification of image origin
    :param entry: Image features and associated metadata used for classification.

    only applies to 20260225_185933:
    flux2 klein fp16, timm/vit-base dino v3 lvd\n
    dim_factor = 3, condense_factor = 2,  top_k = 4, dtype = "float16", alpha = 0.5
    """
    laplace = entry["laplace_mean"]  # Above ~4.2-4.3 for GNE
    sobel = entry["sobel_mean"]  # Below 4 for GNE
    max_ff_mag = entry["max_fourier_magnitude"]  # Above 1500 for GNE
    bce_loss = entry["bce_loss"]  # bce_los around -50 -60 is more likely to be GNE)
    perturbed_bce_loss = entry["perturbed_bce_loss"]  # bce_los around -50 -60 is more likely to be GNE)

    image_mean = entry["image_mean"]

    laplace_metric = lambda x: any(x > 4.25 for x in laplace)
    mean_metric = lambda x: any(y >= 5 for y in x)
    bce_metric = lambda x: 40 < abs(x) < 60
    max_ff_metric = lambda x: x >= 150000
    sobel_metric = lambda x: any(y < 4 for y in x)

    total = [0.8, 0.66, 0.7, 0.66, 0.66]
    score = []

    if laplace_metric(laplace):
        score.append(0)

    if sobel_metric(sobel):
        score.append(1)

    if max_ff_metric(max_ff_mag):
        score.append(2)

    if bce_metric(bce_loss) and bce_metric(perturbed_bce_loss):
        score.append(3)

    if mean_metric(image_mean):
        score.append(4)

    probability = np.sum([total[x] for x in score])
    return probability if probability.item() > 0 else 0


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

    """Map data from [in_min, in_max] to [out_min, out_max]."""


def normalize_to_range(
    data: list[float] | np.ndarray,
    in_min: float,
    in_max: float,
    out_min: float = 0.01,
    out_max: float = 1.0,
) -> np.ndarray:
    """Normalize data to [out_min, out_max] range."""
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return out_min + (data - in_min) * (out_max - out_min) / (in_max - in_min)


def compute_weighted_certainty(
    ae_inference: dict[str, list[float]],
    dc_inference: dict[str, list[float]],
    label: int | None = None,
    ae_low_thresh: float = 0.4,
    ae_high_thresh: float = 0.5,
    dc_low_thresh: float = 0.42,
    dc_high_thresh: float = 0.5,
) -> np.ndarray:
    """
    Compute certainty scores by combining all available inference methods.\n
    Each method contributes a vote (unk: 0=GNE, 1=SYN)). Certainty is the sum normalized 0-5 scale.
    """

    predictor = lambda pct, low_thresh, high_thresh,: "GNE" if pct < low_thresh else "SYN" if pct > high_thresh else "?"
    if label is not None:
        if label == 1:
            header = "SYN [1]"
        else:
            header = "GNE (0)"
        print(f"For : {header} ")

    predictions = [
        {"raw_pred": ae_inference["pred"], "thresh": (ae_low_thresh, ae_high_thresh), "norm": (0.02, 0.90), "norm_pred": None, "result": []},
        {"raw_pred": dc_inference["pred"], "thresh": (dc_low_thresh, dc_high_thresh), "norm": (0.15, 0.80), "norm_pred": None, "result": []},
    ]

    for index in range(len(predictions)):
        predictions[index]["norm_pred"] = normalize_to_range(predictions[index]["raw_pred"], *predictions[index]["norm"])
        predictions[index]["norm_pred"] = predictions[index]["norm_pred"].tolist()
        for image, num in enumerate(predictions[index]["norm_pred"]):
            origin = predictor(num, *predictions[index]["thresh"])
            predictions[index]["result"].append({"index": "ae" if index == 1 else "dc", "img": image, "num": num, "origin": origin})

    result_format = lambda x: f"{x['index']} :{x['origin']} img:{x['img']} " + f"{x['num']:.2%}"
    final_result = []
    final_numeric = []

    for index, result in enumerate(predictions[0]["result"]):
        if predictions[1]["result"][index]["origin"] == result["origin"]:
            most_certain = result
        else:
            low_amount_ae = (predictions[0]["thresh"][0] - result["num"]), result
            high_amount_ae = (result["num"] - predictions[0]["thresh"][1]), result
            low_amount_dc = (predictions[1]["thresh"][0] - predictions[1]["result"][index]["num"]), predictions[1]["result"][1]
            high_amount_dc = (predictions[1]["result"][index]["num"] - predictions[1]["thresh"][1]), predictions[1]["result"][1]
            most_certain = max(
                max(low_amount_ae, high_amount_ae, key=lambda x: x[0]),
                max(low_amount_dc, high_amount_dc, key=lambda x: x[0]),
                key=lambda x: x[0],
            )[1]

        final_numeric.append(most_certain)
        output = result_format(most_certain)
        spacer = " " * (16 - len(output))
        final_result.append(output + spacer)

    thresh = 0.5
    model_pred = (np.array([x["num"] for x in final_numeric]) > thresh).astype(int)
    if label is not None:
        ground_truth = np.full(model_pred.shape, label, dtype=int)
        acc = float(np.mean(model_pred == ground_truth))
        print(f"Model Accuracy: {acc:.2%}")

    pprint(final_result)
    return model_pred
