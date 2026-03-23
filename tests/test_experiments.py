# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
"""Run all feature experiments on Hemg art dataset and compare.

Experiments:
  1. Artwork features only (49 features) — baseline
  2. Style features only (15 features)
  3. Artwork + Style combined (64 features)
  4. CLIP embeddings (768 features)
  5. CLIP + Artwork + Style (832 features)

Each experiment: 4000 samples, 5-fold CV, XGBoost/SVM/MLP.
Generates a comparison PDF.
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import load_dataset, Image as HFImage
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_style import StyleExtract

SEED = 42
N_FOLDS = 5
N_PER_CLASS = 2000
REPO = "Hemg/AI-Generated-vs-Real-Images-Datasets"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_dataset_cached():
    """Load and return the Hemg dataset."""
    print("Loading Hemg dataset...")
    ds = load_dataset(REPO, split="train")
    ds = ds.cast_column("image", HFImage(decode=True, mode="RGB"))
    return ds


def extract_artwork_features(ds, indices) -> np.ndarray:
    """Extract 49 artwork features."""
    extractor = ArtworkExtract()
    features = []
    for idx in tqdm(indices, desc="  Artwork features"):
        try:
            img = ds[int(idx)]["image"]
            if img and isinstance(img, Image.Image):
                features.append(extractor(img))
            else:
                features.append(None)
        except Exception:
            features.append(None)
    df = pd.DataFrame([f for f in features if f is not None]).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    return np.where(np.isfinite(X), X, 0), list(df.columns), [i for i, f in enumerate(features) if f is not None]


def extract_style_features(ds, indices) -> np.ndarray:
    """Extract 15 style features."""
    extractor = StyleExtract()
    features = []
    for idx in tqdm(indices, desc="  Style features"):
        try:
            img = ds[int(idx)]["image"]
            if img and isinstance(img, Image.Image):
                features.append(extractor(img))
            else:
                features.append(None)
        except Exception:
            features.append(None)
    df = pd.DataFrame([f for f in features if f is not None]).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    return np.where(np.isfinite(X), X, 0), list(df.columns), [i for i, f in enumerate(features) if f is not None]


def extract_clip_features(ds, indices) -> np.ndarray:
    """Extract CLIP ViT-B/32 embeddings (512-d)."""
    from transformers import CLIPProcessor, CLIPModel

    print("  Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    features = []
    valid = []
    batch_size = 32

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="  CLIP features"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        images = []
        batch_valid = []
        for i, idx in enumerate(batch_indices):
            try:
                img = ds[int(idx)]["image"]
                if img and isinstance(img, Image.Image):
                    images.append(img)
                    batch_valid.append(batch_start + i)
            except Exception:
                pass

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                embeddings = outputs.cpu().numpy()
            else:
                embeddings = outputs.pooler_output.cpu().numpy()

        features.append(embeddings)
        valid.extend(batch_valid)

    X = np.vstack(features)
    return X, [f"clip_{i}" for i in range(X.shape[1])], valid


def run_cv(X, y, model_type="xgb"):
    """5-fold CV, return metrics dict."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == "xgb":
            spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
            params = {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "scale_pos_weight": spw, "seed": SEED,
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, num_boost_round=300,
                              evals=[(dtest, "test")], early_stopping_rounds=15,
                              verbose_eval=False)
            y_prob = model.predict(dtest)
        elif model_type == "svm":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            svm = SVC(kernel="rbf", probability=True, random_state=SEED)
            svm.fit(X_tr, y_train)
            y_prob = svm.predict_proba(X_te)[:, 1]
        elif model_type == "mlp":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000,
                                random_state=SEED, early_stopping=True)
            mlp.fit(X_tr, y_train)
            y_prob = mlp.predict_proba(X_te)[:, 1]

        all_true.extend(y_test)
        all_prob.extend(y_prob)

    y_true = np.array(all_true)
    y_prob = np.array(all_prob)
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
    }


def generate_pdf(experiments):
    """Generate comparison PDF."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"experiments_comparison_{timestamp}.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # PAGE 1: Title + comparison chart
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")

        fig.suptitle("Feature Experiment Comparison\nfor AI Artwork Detection",
                     fontsize=18, fontweight="bold", fontfamily="serif", y=0.96)
        fig.text(0.5, 0.89, f"negate project — darkshapes — {datetime.now().strftime('%B %d, %Y')}",
                 fontsize=10, ha="center", fontfamily="serif", style="italic")
        fig.text(0.5, 0.86, f"Dataset: Hemg AI-Art vs Real-Art | {N_PER_CLASS*2} samples | 5-fold CV",
                 fontsize=9, ha="center", fontfamily="serif")

        # Grouped bar chart: accuracy by experiment and model
        ax = fig.add_axes([0.1, 0.45, 0.8, 0.35])

        exp_names = [e["name"] for e in experiments]
        n_exp = len(exp_names)
        x = np.arange(n_exp)
        w = 0.25

        for i, (model, color) in enumerate([("xgb", "#4472C4"), ("svm", "#ED7D31"), ("mlp", "#70AD47")]):
            accs = [e["results"][model]["accuracy"] for e in experiments]
            bars = ax.bar(x + i * w - w, accs, w, label=model.upper(), color=color)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f"{bar.get_height():.1%}", ha="center", fontsize=6.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([e["short_name"] for e in experiments], fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title("Accuracy by Feature Set and Model", fontsize=12, fontfamily="serif")
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)

        # Summary table
        ax_table = fig.add_axes([0.05, 0.08, 0.9, 0.3])
        ax_table.axis("off")

        table_data = []
        for e in experiments:
            best_model = max(e["results"], key=lambda m: e["results"][m]["accuracy"])
            best = e["results"][best_model]
            table_data.append([
                e["short_name"],
                str(e["n_features"]),
                f"{best['accuracy']:.2%}",
                f"{best['precision']:.2%}",
                f"{best['recall']:.2%}",
                f"{best['roc_auc']:.4f}",
                best_model.upper(),
                e.get("extract_time", "?"),
            ])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Features", "Count", "Best Acc", "Prec", "Recall", "AUC", "Model", "Time"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2: ROC curves
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle("ROC Curves by Experiment (Best Model)", fontsize=14,
                     fontweight="bold", fontfamily="serif", y=0.96)

        colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#9B59B6"]
        ax = fig.add_axes([0.12, 0.5, 0.76, 0.38])

        for i, e in enumerate(experiments):
            best_model = max(e["results"], key=lambda m: e["results"][m]["roc_auc"])
            r = e["results"][best_model]
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                   label=f"{e['short_name']} (AUC={r['roc_auc']:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        # Analysis text
        ax_text = fig.add_axes([0.08, 0.05, 0.84, 0.38])
        ax_text.axis("off")

        # Find best and worst
        best_exp = max(experiments, key=lambda e: max(e["results"][m]["accuracy"] for m in e["results"]))
        worst_exp = min(experiments, key=lambda e: max(e["results"][m]["accuracy"] for m in e["results"]))
        best_acc = max(best_exp["results"][m]["accuracy"] for m in best_exp["results"])
        worst_acc = max(worst_exp["results"][m]["accuracy"] for m in worst_exp["results"])

        analysis = (
            "Analysis\n\n"
            f"Best performing: {best_exp['name']} at {best_acc:.1%}\n"
            f"Worst performing: {worst_exp['name']} at {worst_acc:.1%}\n"
            f"Improvement from best to worst: {(best_acc - worst_acc)*100:+.1f}pp\n\n"
        )

        # Check if CLIP exists
        clip_exp = [e for e in experiments if "clip" in e["short_name"].lower()]
        art_exp = [e for e in experiments if e["short_name"] == "Artwork (49)"]

        if clip_exp and art_exp:
            clip_acc = max(clip_exp[0]["results"][m]["accuracy"] for m in clip_exp[0]["results"])
            art_acc = max(art_exp[0]["results"][m]["accuracy"] for m in art_exp[0]["results"])
            analysis += (
                f"CLIP vs hand-crafted: {clip_acc:.1%} vs {art_acc:.1%} "
                f"({(clip_acc - art_acc)*100:+.1f}pp)\n"
            )
            if clip_acc > art_acc + 0.03:
                analysis += "Learned features significantly outperform hand-crafted features.\n"
            elif clip_acc < art_acc - 0.03:
                analysis += "Surprisingly, hand-crafted features outperform CLIP on this task.\n"
            else:
                analysis += "Learned and hand-crafted features perform similarly.\n"

        # Check if combined helps
        combined_exp = [e for e in experiments if "+" in e["short_name"]]
        if combined_exp:
            comb_acc = max(combined_exp[-1]["results"][m]["accuracy"] for m in combined_exp[-1]["results"])
            analysis += (
                f"\nCombined features: {comb_acc:.1%}\n"
            )
            if comb_acc > best_acc - 0.01:
                analysis += "Combining features achieves the best overall performance.\n"
            else:
                analysis += "Combining features does not improve over the best individual set.\n"

        analysis += (
            "\nConclusions\n\n"
            "This comparison tests whether:\n"
            "  1. Style-specific craft features add signal beyond generic statistics\n"
            "  2. Learned representations (CLIP) outperform hand-crafted features\n"
            "  3. Combining multiple feature types improves detection\n\n"
            "All experiments use the same dataset (Hemg AI Art vs Real Art),\n"
            "same sample size, and same evaluation methodology.\n"
        )

        ax_text.text(0, 1, analysis, transform=ax_text.transAxes, fontsize=9,
                    ha="left", va="top", fontfamily="serif")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to: {pdf_path}")
    return pdf_path


def main():
    print("=" * 60)
    print("  FEATURE EXPERIMENTS COMPARISON")
    print("  Dataset: Hemg AI Art vs Real Art")
    print(f"  Samples: {N_PER_CLASS * 2} ({N_PER_CLASS} per class)")
    print("=" * 60)

    ds = load_dataset_cached()
    all_labels = ds["label"]

    # Select balanced indices
    rng = np.random.RandomState(SEED)
    idx_0 = [i for i, l in enumerate(all_labels) if l == 0]
    idx_1 = [i for i, l in enumerate(all_labels) if l == 1]
    chosen_0 = rng.choice(idx_0, size=N_PER_CLASS, replace=False)
    chosen_1 = rng.choice(idx_1, size=N_PER_CLASS, replace=False)
    all_indices = np.concatenate([chosen_0, chosen_1])
    # Labels: 0=AI(synthetic), 1=Real(genuine) in dataset
    # We want: 0=genuine, 1=synthetic
    y = np.array([1] * N_PER_CLASS + [0] * N_PER_CLASS)

    experiments = []

    # === Experiment 1: Artwork features (49) ===
    print("\n" + "=" * 50)
    print("  Experiment 1: Artwork Features (49)")
    print("=" * 50)
    t0 = time.time()
    X_art, art_names, art_valid = extract_artwork_features(ds, all_indices)
    t_art = f"{time.time() - t0:.0f}s"
    y_art = y[art_valid]
    print(f"  {X_art.shape[0]} images, {X_art.shape[1]} features, {t_art}")

    exp1 = {"name": "Artwork Features (Li & Stamp + FFT/DCT)", "short_name": "Artwork (49)",
            "n_features": X_art.shape[1], "extract_time": t_art, "results": {}}
    for model in ["xgb", "svm", "mlp"]:
        print(f"  {model.upper()}...")
        exp1["results"][model] = run_cv(X_art, y_art, model)
        print(f"    acc={exp1['results'][model]['accuracy']:.2%}")
    experiments.append(exp1)

    # === Experiment 2: Style features (15) ===
    print("\n" + "=" * 50)
    print("  Experiment 2: Style Features (15)")
    print("=" * 50)
    t0 = time.time()
    X_style, style_names, style_valid = extract_style_features(ds, all_indices)
    t_style = f"{time.time() - t0:.0f}s"
    y_style = y[style_valid]
    print(f"  {X_style.shape[0]} images, {X_style.shape[1]} features, {t_style}")

    exp2 = {"name": "Style Features (stroke/palette/composition/texture)", "short_name": "Style (15)",
            "n_features": X_style.shape[1], "extract_time": t_style, "results": {}}
    for model in ["xgb", "svm", "mlp"]:
        print(f"  {model.upper()}...")
        exp2["results"][model] = run_cv(X_style, y_style, model)
        print(f"    acc={exp2['results'][model]['accuracy']:.2%}")
    experiments.append(exp2)

    # === Experiment 3: Artwork + Style combined (64) ===
    print("\n" + "=" * 50)
    print("  Experiment 3: Artwork + Style Combined (64)")
    print("=" * 50)
    # Align valid indices
    common_valid = sorted(set(art_valid) & set(style_valid))
    art_mask = [art_valid.index(v) for v in common_valid]
    style_mask = [style_valid.index(v) for v in common_valid]
    X_combined = np.hstack([X_art[art_mask], X_style[style_mask]])
    y_combined = y[common_valid]
    print(f"  {X_combined.shape[0]} images, {X_combined.shape[1]} features")

    exp3 = {"name": "Artwork + Style Combined", "short_name": "Art+Style (64)",
            "n_features": X_combined.shape[1], "extract_time": "combined", "results": {}}
    for model in ["xgb", "svm", "mlp"]:
        print(f"  {model.upper()}...")
        exp3["results"][model] = run_cv(X_combined, y_combined, model)
        print(f"    acc={exp3['results'][model]['accuracy']:.2%}")
    experiments.append(exp3)

    # === Experiment 4: CLIP embeddings (512) ===
    print("\n" + "=" * 50)
    print("  Experiment 4: CLIP ViT-B/32 Embeddings (512)")
    print("=" * 50)
    t0 = time.time()
    X_clip, clip_names, clip_valid = extract_clip_features(ds, all_indices)
    t_clip = f"{time.time() - t0:.0f}s"
    y_clip = y[clip_valid]
    print(f"  {X_clip.shape[0]} images, {X_clip.shape[1]} features, {t_clip}")

    exp4 = {"name": "CLIP ViT-B/32 Embeddings", "short_name": "CLIP (512)",
            "n_features": X_clip.shape[1], "extract_time": t_clip, "results": {}}
    for model in ["xgb", "svm", "mlp"]:
        print(f"  {model.upper()}...")
        exp4["results"][model] = run_cv(X_clip, y_clip, model)
        print(f"    acc={exp4['results'][model]['accuracy']:.2%}")
    experiments.append(exp4)

    # === Experiment 5: CLIP + Artwork + Style (all combined) ===
    print("\n" + "=" * 50)
    print("  Experiment 5: CLIP + Artwork + Style (all)")
    print("=" * 50)
    common_all = sorted(set(art_valid) & set(style_valid) & set(clip_valid))
    art_m = [art_valid.index(v) for v in common_all]
    style_m = [style_valid.index(v) for v in common_all]
    clip_m = [clip_valid.index(v) for v in common_all]
    X_all = np.hstack([X_art[art_m], X_style[style_m], X_clip[clip_m]])
    y_all = y[common_all]
    print(f"  {X_all.shape[0]} images, {X_all.shape[1]} features")

    exp5 = {"name": "CLIP + Artwork + Style (Everything)", "short_name": "All Combined",
            "n_features": X_all.shape[1], "extract_time": "combined", "results": {}}
    for model in ["xgb", "svm", "mlp"]:
        print(f"  {model.upper()}...")
        exp5["results"][model] = run_cv(X_all, y_all, model)
        print(f"    acc={exp5['results'][model]['accuracy']:.2%}")
    experiments.append(exp5)

    # Save results (without y_true/y_prob arrays for JSON)
    json_results = []
    for e in experiments:
        je = {k: v for k, v in e.items() if k != "results"}
        je["results"] = {}
        for m, r in e["results"].items():
            je["results"][m] = {k: v for k, v in r.items() if k not in ("y_true", "y_prob")}
        json_results.append(je)

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "experiments_results.json"
    with open(json_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "experiments": json_results}, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Generate PDF
    print("\nGenerating comparison PDF...")
    generate_pdf(experiments)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    for e in experiments:
        best_model = max(e["results"], key=lambda m: e["results"][m]["accuracy"])
        best = e["results"][best_model]
        print(f"  {e['short_name']:20s}  acc={best['accuracy']:.2%}  auc={best['roc_auc']:.4f}  ({best_model})")


if __name__ == "__main__":
    main()
