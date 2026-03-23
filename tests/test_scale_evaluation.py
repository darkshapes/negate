# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
"""Scale evaluation: test if more training data improves artwork detection.

Runs the 49-feature pipeline on increasing sample sizes from Hemg (art vs art)
to determine if 71% accuracy is a data problem or a feature problem.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import xgboost as xgb
from datasets import load_dataset, Image as HFImage
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract

SEED = 42
N_FOLDS = 5
REPO = "Hemg/AI-Generated-vs-Real-Images-Datasets"
SAMPLE_SIZES = [400, 1000, 2000, 4000]  # total (half per class)
RESULTS_DIR = Path(__file__).parent.parent / "results"


def extract_features_cached(dataset, n_per_class: int, extractor: ArtworkExtract):
    """Extract features, balanced per class."""
    all_labels = dataset["label"]
    features, labels, errors = [], [], 0

    rng = np.random.RandomState(SEED)

    for lbl in [0, 1]:
        indices = [i for i, l in enumerate(all_labels) if l == lbl]
        chosen = rng.choice(indices, size=min(n_per_class, len(indices)), replace=False)

        for idx in tqdm(chosen, desc=f"  Label {lbl} (n={n_per_class})"):
            try:
                img = dataset[int(idx)]["image"]
                if img is None or not isinstance(img, Image.Image):
                    errors += 1
                    continue
                feat = extractor(img)
                features.append(feat)
                # label 0 = AI art (synthetic), label 1 = Real art (genuine)
                # We want: 0 = genuine, 1 = synthetic
                labels.append(1 if lbl == 0 else 0)
            except Exception:
                errors += 1

    print(f"  Extracted {len(features)} ({errors} errors)")
    df = pd.DataFrame(features).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0)
    y = np.array(labels)
    return X, y, list(df.columns)


def run_cv(X, y, model_type="xgb"):
    """Run 5-fold CV, return pooled y_true, y_prob."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == "xgb":
            spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
            params = {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "scale_pos_weight": spw, "seed": SEED,
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, num_boost_round=200,
                              evals=[(dtest, "test")], early_stopping_rounds=10,
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
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
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
    }


def generate_pdf(all_results):
    """Generate scaling analysis PDF."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"scale_evaluation_{timestamp}.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # PAGE 1: Title + scaling curves
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")

        fig.suptitle("Scaling Analysis: Does More Data Improve\nArtwork Detection Accuracy?",
                     fontsize=16, fontweight="bold", fontfamily="serif", y=0.96)

        # Subtitle
        fig.text(0.5, 0.90, f"negate project — darkshapes — {datetime.now().strftime('%B %d, %Y')}",
                 fontsize=10, ha="center", fontfamily="serif", style="italic")

        fig.text(0.5, 0.87, "Dataset: Hemg/AI-Generated-vs-Real-Images-Datasets (AI Art vs Real Art)",
                 fontsize=9, ha="center", fontfamily="serif")

        # Accuracy scaling curve
        ax1 = fig.add_axes([0.12, 0.52, 0.76, 0.3])
        sizes = [r["total"] for r in all_results]

        for model, color, marker in [("xgb", "#4472C4", "o"), ("svm", "#ED7D31", "s"), ("mlp", "#70AD47", "^")]:
            accs = [r[model]["accuracy"] for r in all_results]
            ax1.plot(sizes, accs, f"-{marker}", color=color, label=model.upper(), markersize=8, linewidth=2)
            for x, y in zip(sizes, accs):
                ax1.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)

        ax1.set_xlabel("Total Training Samples", fontsize=10)
        ax1.set_ylabel("5-Fold CV Accuracy", fontsize=10)
        ax1.set_title("Accuracy vs Training Set Size", fontsize=12, fontfamily="serif")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 1.0)
        ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random chance")

        # AUC scaling curve
        ax2 = fig.add_axes([0.12, 0.12, 0.76, 0.3])

        for model, color, marker in [("xgb", "#4472C4", "o"), ("svm", "#ED7D31", "s"), ("mlp", "#70AD47", "^")]:
            aucs = [r[model]["roc_auc"] for r in all_results]
            ax2.plot(sizes, aucs, f"-{marker}", color=color, label=model.upper(), markersize=8, linewidth=2)
            for x, y in zip(sizes, aucs):
                ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)

        ax2.set_xlabel("Total Training Samples", fontsize=10)
        ax2.set_ylabel("5-Fold CV ROC-AUC", fontsize=10)
        ax2.set_title("ROC-AUC vs Training Set Size", fontsize=12, fontfamily="serif")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.0)

        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2: Results table + analysis
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle("Detailed Results & Analysis", fontsize=14,
                     fontweight="bold", fontfamily="serif", y=0.96)

        # Results table
        ax_table = fig.add_axes([0.05, 0.62, 0.9, 0.28])
        ax_table.axis("off")

        table_data = []
        for r in all_results:
            for model in ["xgb", "svm", "mlp"]:
                m = r[model]
                table_data.append([
                    str(r["total"]), model.upper(),
                    f"{m['accuracy']:.2%}", f"{m['precision']:.2%}",
                    f"{m['recall']:.2%}", f"{m['f1']:.2%}", f"{m['roc_auc']:.4f}"
                ])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Samples", "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.3)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

        # Analysis
        ax_text = fig.add_axes([0.08, 0.05, 0.84, 0.52])
        ax_text.axis("off")

        best_final = max(all_results[-1]["xgb"]["accuracy"],
                        all_results[-1]["svm"]["accuracy"],
                        all_results[-1]["mlp"]["accuracy"])
        best_initial = max(all_results[0]["xgb"]["accuracy"],
                          all_results[0]["svm"]["accuracy"],
                          all_results[0]["mlp"]["accuracy"])
        improvement = best_final - best_initial

        analysis = (
            "Analysis\n\n"
            f"Sample sizes tested: {', '.join(str(r['total']) for r in all_results)}\n"
            f"Best accuracy at smallest size ({all_results[0]['total']}): {best_initial:.1%}\n"
            f"Best accuracy at largest size ({all_results[-1]['total']}): {best_final:.1%}\n"
            f"Improvement from scaling: {improvement:+.1%}pp\n\n"
        )

        if improvement > 0.10:
            analysis += (
                "FINDING: Significant improvement with more data.\n"
                "The 49 features have capacity to learn — the initial low accuracy was\n"
                "primarily a data limitation. With sufficient training data, the hand-crafted\n"
                "features can achieve useful detection rates on artwork.\n\n"
                "Recommendation: Scale to even larger samples (10K+) and consider\n"
                "integrating these features into the negate pipeline."
            )
        elif improvement > 0.03:
            analysis += (
                "FINDING: Modest improvement with more data.\n"
                "More data helps somewhat, but accuracy is plateauing. The features\n"
                "capture some genuine signal but are limited by their expressiveness.\n\n"
                "Recommendation: The hand-crafted features are hitting a ceiling.\n"
                "To push past this, the pipeline needs learned features — either\n"
                "fine-tuned CLIP/DINOv2 or the self-supervised approach from\n"
                "Zhong et al. (2026)."
            )
        else:
            analysis += (
                "FINDING: Minimal improvement with more data.\n"
                "The features are saturated — adding more training data does not help.\n"
                "The 49 hand-crafted features simply don't capture enough discriminative\n"
                "information to distinguish AI art from human art.\n\n"
                "Recommendation: Fundamentally different features are needed.\n"
                "Hand-crafted statistics cannot match the representational power of\n"
                "learned features for this task."
            )

        analysis += (
            "\n\nContext\n\n"
            "This evaluation uses only the Hemg dataset where BOTH classes are artwork.\n"
            "This is the hardest and most honest test — no content shortcuts.\n"
            "All processing is CPU-only, 49 features per image.\n"
            "5-fold stratified cross-validation with fixed random seed (42).\n"
        )

        ax_text.text(0, 1, analysis, transform=ax_text.transAxes, fontsize=9,
                    ha="left", va="top", fontfamily="serif")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to: {pdf_path}")
    return pdf_path


def main():
    print("=" * 60)
    print("  SCALING ANALYSIS: Art Detection vs Training Data Size")
    print("  Dataset: Hemg (AI Art vs Real Art)")
    print("=" * 60)

    print("\nLoading dataset...")
    ds = load_dataset(REPO, split="train")
    ds = ds.cast_column("image", HFImage(decode=True, mode="RGB"))
    print(f"  Total rows: {len(ds)}")

    extractor = ArtworkExtract()
    all_results = []

    # We extract at the largest size once, then subsample
    max_per_class = max(SAMPLE_SIZES) // 2
    print(f"\nExtracting features for {max_per_class} per class...")
    X_full, y_full, feature_names = extract_features_cached(ds, max_per_class, extractor)
    print(f"  Total: {len(y_full)} images, {X_full.shape[1]} features")
    print(f"  Balance: {np.sum(y_full==0)} genuine, {np.sum(y_full==1)} synthetic")

    for total in SAMPLE_SIZES:
        per_class = total // 2
        print(f"\n{'='*40}")
        print(f"  Testing with {total} samples ({per_class} per class)")
        print(f"{'='*40}")

        # Subsample from the full extraction
        rng = np.random.RandomState(SEED)
        idx_0 = np.where(y_full == 0)[0]
        idx_1 = np.where(y_full == 1)[0]
        chosen_0 = rng.choice(idx_0, size=min(per_class, len(idx_0)), replace=False)
        chosen_1 = rng.choice(idx_1, size=min(per_class, len(idx_1)), replace=False)
        chosen = np.concatenate([chosen_0, chosen_1])
        X = X_full[chosen]
        y = y_full[chosen]

        result = {"total": len(y)}
        for model in ["xgb", "svm", "mlp"]:
            print(f"  Running {model.upper()}...")
            result[model] = run_cv(X, y, model)
            print(f"    acc={result[model]['accuracy']:.2%} auc={result[model]['roc_auc']:.4f}")

        all_results.append(result)

    # Save JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "scale_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": REPO,
            "feature_count": X_full.shape[1],
            "results": all_results,
        }, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Generate PDF
    print("\nGenerating PDF...")
    generate_pdf(all_results)

    # Print summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        best = max(r["xgb"]["accuracy"], r["svm"]["accuracy"], r["mlp"]["accuracy"])
        print(f"  n={r['total']:5d}  best_acc={best:.2%}")


if __name__ == "__main__":
    main()
