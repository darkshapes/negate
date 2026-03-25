# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
"""CLIP bias test on Defactify MS-COCOAI dataset.

Tests whether CLIP's detection advantage comes from recognizing its own
latent fingerprint in images from CLIP-based generators.

Dataset: Rajarshi-Roy-research/Defactify_Image_Dataset (96K images)
- Label_B=0: Real (MS COCO)
- Label_B=1: SD 2.1 (uses CLIP)
- Label_B=2: SDXL (uses CLIP-L + CLIP-G)
- Label_B=3: SD 3 (uses CLIP-L + CLIP-G + T5)
- Label_B=4: Midjourney v6 (proprietary, unknown)
- Label_B=5: DALL-E 3 (uses T5, NOT CLIP)

Key comparison: CLIP accuracy on SD 2.1/SDXL (pure CLIP) vs DALL-E 3 (no CLIP).
If CLIP's advantage is larger on CLIP-based generators, bias is confirmed.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import load_dataset, Image as HFImage
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_style import StyleExtract

SEED = 42
N_FOLDS = 5
N_PER_CLASS = 500  # per generator
RESULTS_DIR = Path(__file__).parent.parent / "results"

GENERATORS = {
    0: {"name": "Real (MS COCO)", "uses_clip": None},
    1: {"name": "SD 2.1", "uses_clip": True},
    2: {"name": "SDXL", "uses_clip": True},
    3: {"name": "SD 3", "uses_clip": True},  # hybrid: CLIP + T5
    4: {"name": "Midjourney v6", "uses_clip": "unknown"},
    5: {"name": "DALL-E 3", "uses_clip": False},  # T5 only
}


def extract_handcrafted(images):
    art = ArtworkExtract()
    style = StyleExtract()
    features = []
    for img in tqdm(images, desc="  Hand-crafted"):
        try:
            f = art(img)
            f |= style(img)
            features.append(f)
        except Exception:
            features.append(None)
    valid = [i for i, f in enumerate(features) if f is not None]
    df = pd.DataFrame([f for f in features if f is not None]).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    return np.where(np.isfinite(X), X, 0), valid


def extract_clip(images):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_emb, valid = [], []
    bs = 32
    for i in tqdm(range(0, len(images), bs), desc="  CLIP"):
        batch = [img for img in images[i:i+bs] if img and isinstance(img, Image.Image)]
        batch_idx = [i+j for j, img in enumerate(images[i:i+bs]) if img and isinstance(img, Image.Image)]
        if not batch:
            continue
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            out = model.get_image_features(**inputs)
            emb = out.pooler_output.cpu().numpy() if hasattr(out, 'pooler_output') else out.cpu().numpy()
        all_emb.append(emb)
        valid.extend(batch_idx)
    return np.vstack(all_emb), valid


def run_cv(X, y, model_type="svm"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if model_type == "svm":
            scaler = StandardScaler()
            clf = SVC(kernel="rbf", probability=True, random_state=SEED)
            clf.fit(scaler.fit_transform(X_train), y_train)
            y_prob = clf.predict_proba(scaler.transform(X_test))[:, 1]
        elif model_type == "xgb":
            spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            m = xgb.train({"objective": "binary:logistic", "max_depth": 5,
                           "learning_rate": 0.1, "scale_pos_weight": spw, "seed": SEED},
                          dtrain, num_boost_round=200, evals=[(dtest, "t")],
                          early_stopping_rounds=10, verbose_eval=False)
            y_prob = m.predict(dtest)
        all_true.extend(y_test)
        all_prob.extend(y_prob)

    yt, yp = np.array(all_true), np.array(all_prob)
    ypr = (yp > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(yt, ypr)),
        "precision": float(precision_score(yt, ypr, zero_division=0)),
        "recall": float(recall_score(yt, ypr, zero_division=0)),
        "roc_auc": float(roc_auc_score(yt, yp)),
    }


def generate_pdf(results):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"clip_bias_defactify_{ts}.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.93, "CLIP Bias Analysis:\nDefactify MS-COCOAI Dataset",
                transform=ax.transAxes, fontsize=18, fontweight="bold",
                ha="center", va="top", fontfamily="serif")
        ax.text(0.5, 0.83, f"negate — darkshapes — {datetime.now().strftime('%B %d, %Y')}",
                transform=ax.transAxes, fontsize=10, ha="center", fontfamily="serif", style="italic")

        hyp = (
            "Hypothesis: CLIP embeddings achieve high detection accuracy because many\n"
            "generators use CLIP as their text encoder, so CLIP recognizes its own fingerprint.\n\n"
            "Test: Compare CLIP vs hand-crafted feature accuracy PER GENERATOR.\n"
            "If CLIP's advantage is larger on CLIP-based generators (SD 2.1, SDXL, SD 3)\n"
            "than on non-CLIP generators (DALL-E 3), bias is confirmed."
        )
        ax.text(0.08, 0.74, hyp, transform=ax.transAxes, fontsize=9,
                ha="left", va="top", fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

        # Results table
        table_data = []
        for r in results:
            table_data.append([
                r["generator"], "Yes" if r["uses_clip"] is True else "No" if r["uses_clip"] is False else "?",
                f"{r['handcrafted_best']:.1%}", f"{r['clip_best']:.1%}",
                f"{r['clip_best'] - r['handcrafted_best']:+.1%}pp"
            ])

        ax_t = fig.add_axes([0.05, 0.38, 0.9, 0.28])
        ax_t.axis("off")
        table = ax_t.table(cellText=table_data,
                           colLabels=["Generator", "Uses CLIP?", "Hand-crafted", "CLIP", "CLIP Advantage"],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

        # Analysis
        clip_gens = [r for r in results if r["uses_clip"] is True]
        non_clip = [r for r in results if r["uses_clip"] is False]
        clip_avg_adv = np.mean([r["clip_best"] - r["handcrafted_best"] for r in clip_gens]) if clip_gens else 0
        non_clip_avg_adv = np.mean([r["clip_best"] - r["handcrafted_best"] for r in non_clip]) if non_clip else 0

        if clip_gens and non_clip:
            diff = clip_avg_adv - non_clip_avg_adv
            if diff > 0.05:
                verdict = (
                    f"CLIP BIAS CONFIRMED.\n\n"
                    f"CLIP advantage on CLIP-based generators: {clip_avg_adv:+.1%}pp (avg)\n"
                    f"CLIP advantage on non-CLIP generators:   {non_clip_avg_adv:+.1%}pp (avg)\n"
                    f"Difference: {diff:+.1%}pp\n\n"
                    "CLIP performs significantly better on images from generators that use\n"
                    "CLIP internally. This suggests CLIP partially recognizes its own latent\n"
                    "fingerprint rather than detecting universal generation artifacts."
                )
            elif diff < -0.05:
                verdict = (
                    f"CLIP BIAS NOT CONFIRMED (reverse pattern).\n\n"
                    f"CLIP advantage on CLIP-based generators: {clip_avg_adv:+.1%}pp (avg)\n"
                    f"CLIP advantage on non-CLIP generators:   {non_clip_avg_adv:+.1%}pp (avg)\n\n"
                    "CLIP actually has a LARGER advantage on non-CLIP generators.\n"
                    "This suggests CLIP detects genuine visual artifacts, not its own fingerprint."
                )
            else:
                verdict = (
                    f"NO SIGNIFICANT CLIP BIAS.\n\n"
                    f"CLIP advantage on CLIP-based generators: {clip_avg_adv:+.1%}pp (avg)\n"
                    f"CLIP advantage on non-CLIP generators:   {non_clip_avg_adv:+.1%}pp (avg)\n"
                    f"Difference: {diff:+.1%}pp (not significant)\n\n"
                    "CLIP's advantage is consistent across generator types, suggesting\n"
                    "it detects genuine visual differences, not architectural fingerprints."
                )
        else:
            verdict = "Insufficient data."

        ax.text(0.08, 0.3, verdict, transform=ax.transAxes, fontsize=9.5,
                ha="left", va="top", fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#66BB6A"))

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to: {pdf_path}")
    return pdf_path


def main():
    print("=" * 60)
    print("  CLIP BIAS ANALYSIS — Defactify MS-COCOAI")
    print("  5 generators, labeled, semantically matched")
    print("=" * 60)

    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", split="train")
    ds = ds.cast_column("Image", HFImage(decode=True, mode="RGB"))
    print(f"Total: {len(ds)} images")

    rng = np.random.RandomState(SEED)
    real_indices = [i for i, l in enumerate(ds["Label_B"]) if l == 0]
    real_sample = rng.choice(real_indices, size=N_PER_CLASS, replace=False)
    real_images = [ds[int(i)]["Image"] for i in tqdm(real_sample, desc="Loading real")]

    all_results = []

    for gen_id in [1, 2, 3, 4, 5]:
        gen_info = GENERATORS[gen_id]
        print(f"\n{'='*50}")
        print(f"  {gen_info['name']} (uses_clip={gen_info['uses_clip']}) vs Real")
        print(f"{'='*50}")

        gen_indices = [i for i, l in enumerate(ds["Label_B"]) if l == gen_id]
        gen_sample = rng.choice(gen_indices, size=N_PER_CLASS, replace=False)
        gen_images = [ds[int(i)]["Image"] for i in tqdm(gen_sample, desc=f"Loading {gen_info['name']}")]

        all_images = real_images + gen_images
        y = np.array([0] * len(real_images) + [1] * len(gen_images))

        # Hand-crafted
        print("  Extracting hand-crafted features...")
        X_hc, hc_valid = extract_handcrafted(all_images)
        y_hc = y[hc_valid]

        # CLIP
        print("  Extracting CLIP features...")
        X_clip, clip_valid = extract_clip(all_images)
        y_clip = y[clip_valid]

        result = {"generator": gen_info["name"], "uses_clip": gen_info["uses_clip"]}

        for feat_name, X_f, y_f in [("handcrafted", X_hc, y_hc), ("clip", X_clip, y_clip)]:
            for model in ["xgb", "svm"]:
                key = f"{feat_name}_{model}"
                r = run_cv(X_f, y_f, model)
                result[key] = r
                print(f"    {key:25s} acc={r['accuracy']:.2%} auc={r['roc_auc']:.4f}")

        result["handcrafted_best"] = max(result["handcrafted_xgb"]["accuracy"],
                                         result["handcrafted_svm"]["accuracy"])
        result["clip_best"] = max(result["clip_xgb"]["accuracy"],
                                  result["clip_svm"]["accuracy"])
        result["clip_advantage"] = result["clip_best"] - result["handcrafted_best"]
        all_results.append(result)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "clip_bias_defactify_results.json"
    with open(json_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)

    generate_pdf(all_results)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY — CLIP advantage per generator")
    print(f"{'='*60}")
    for r in all_results:
        clip_tag = "CLIP" if r["uses_clip"] is True else "NO-CLIP" if r["uses_clip"] is False else "???"
        print(f"  {r['generator']:20s} [{clip_tag:7s}]  hand={r['handcrafted_best']:.1%}  "
              f"clip={r['clip_best']:.1%}  delta={r['clip_advantage']:+.1%}")

    clip_gens = [r for r in all_results if r["uses_clip"] is True]
    non_clip = [r for r in all_results if r["uses_clip"] is False]
    if clip_gens and non_clip:
        print(f"\n  Avg CLIP advantage on CLIP generators: "
              f"{np.mean([r['clip_advantage'] for r in clip_gens]):+.1%}")
        print(f"  Avg CLIP advantage on non-CLIP generators: "
              f"{np.mean([r['clip_advantage'] for r in non_clip]):+.1%}")


if __name__ == "__main__":
    main()
