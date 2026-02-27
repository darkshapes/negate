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

from typing import Any
import numpy as np


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


def compute_weighted_certainty(
    ae_inference: dict[str, list[float]],
    dc_inference: dict[str, list[float]],
    label: int | None = None,
    dc_syn_thresh: float = 0.4,
    ae_syn_thresh: float = 0.5,
) -> None:
    """
    Compute certainty scores by combining all available inference methods.\n
    Each method contributes a vote (unk: 0=GNE, 1=SYN)). Certainty is the sum normalized 0-5 scale.
    """
    from pprint import pprint

    print(f"For : {'SYN' + ' [1]' if label == 1 else 'GNE' + ' (0)'} ")
    model_thresh = 0.5
    # hstc_thresh = 0
    predictions = {
        "ae_model": (np.array(ae_inference["unk"]) > model_thresh).astype(int),
        "dc_model": (np.array(dc_inference["unk"]) > model_thresh).astype(int),
    }

    pred = {}
    for id_name, model_pred in predictions.items():
        if label is not None:
            ground_truth = np.full(model_pred.shape, label, dtype=int)
            acc = float(np.mean(model_pred == ground_truth))
            print(f"{id_name} Accuracy: {acc:.2%}")
        pred[id_name] = [f"{index} SYN % : {pred:.2%}" for index, pred in enumerate(dc_inference["unk"])]

    pprint(pred)

    # hstc_pred = {
    #     "ae_hstc": (np.array(ae_inference["ae_gne"]) > hstc_thresh).astype(int),
    #     "dc_hstc": (np.array(dc_inference["dc_gne"]) > hstc_thresh).astype(int),
    # }
    # for id_name, hstc in hstc_pred.items():
    #     pred[id_name] = [f"{index} GNE % : {pred:.2%}" for index, pred in enumerate(hstc)]

    # n_images = min(len(ae_inference["unk"]), len(dc_inference["unk"]))
    # inferences = []
    # print(f"ae_inference: {ae_inference}")
    # print(f"dc_inference: {dc_inference}")
    # for i in range(n_images):
    #     syn_weight = dc_inference["dc_syn"][i]
    #     gne_weight = ae_inference["ae_gne"][i], dc_inference["dc_gne"][i]
    #     print(f"image {i} SYN % : {syn_weight}")
    #     print(f"image {i} GNE % : {gne_weight}")
    # unk_weight: float = ae_inference["unk"][i]
    #     candidates = [
    #         (syn_weight, syn_weight[1] * syn_heur_weight),
    #         (gne_weight, gne_weight[1] * gne_heur_weight),
    #         (unk_weight, unk_weight[1] * unk_model_weight),
    #     ]
    #     estimation, _ = max(candidates, key=lambda x: x[1])
    #     print(estimation)
    #     inferences.append((estimation))
    # return inferences

    # model_pred = model_accuracy(model_pred)
    # if context.verbose:
    #     print(f"""          Decision Tree Model result: {model_pred}
    #         SYN Probability (DC VAE): {heur_dc_pred}
    #         GNE Probability (AE VAE): {heur_ae_pred}""")

    # if args.label is not None:
    #     label = "SYN" if args.label == 1 else "GNE"
    #     print(f"For : {label + ' [1]' if args.label == 1 else label + ' (0)'} ")
    #     count = sum(1 for item in inferences if label in item[0])
    #     print(f"{count} / {len(inferences)} {(count / len(inferences)):.2%}")
    # result_dc = []
    # result_ae = []
    # print(ae_inference["ae_gne"])
    # print(dc_inference["dc_gne"])
    # # for index, image in enumerate(ae_inference["unk"]):
    # #     result_dc.append(np.round(dc_inference["unk"][index] - ae_inference["ae_gne"][index]))
    # #     result_ae.append(np.round(image - ae_inference["ae_gne"][index]))
    # print(ae_inference["unk"])
    # # print(dc_inference["unk"])
    # print(result_dc)
    # print(result_ae)
    # thresh = 0.5

    # ae_data = []
    # dc_data = []
    # for index, pred in enumerate(ae_inference["unk"]):
    #     ae_data.append(f"{index} SYN % : {pred:.2%}")
