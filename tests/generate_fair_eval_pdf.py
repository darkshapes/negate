# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
"""Generate PDF report for fair evaluation results.

Reads results/fair_evaluation_results.json and generates a timestamped PDF
with cross-validation metrics, comparison tables, and analysis.
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
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_pdf(results_path: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"fair_evaluation_{timestamp}.pdf"

    with open(results_path) as f:
        data = json.load(f)

    datasets = data["datasets"]

    with PdfPages(str(pdf_path)) as pdf:
        # ===== PAGE 1: Title & Summary =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.88, "Fair Evaluation Report:\n49-Feature Artwork Detection",
                transform=ax.transAxes, fontsize=20, fontweight="bold",
                ha="center", va="top", fontfamily="serif")

        ax.text(0.5, 0.74, f"negate project — darkshapes\n{datetime.now().strftime('%B %d, %Y')}",
                transform=ax.transAxes, fontsize=11, ha="center", va="top",
                fontfamily="serif", style="italic")

        # Why this evaluation matters
        rationale = (
            "Why This Evaluation Matters\n\n"
            "Previous benchmarks used datasets where AI and genuine images had different\n"
            "subject matter (cats vs bananas, WikiArt paintings vs generated illustrations).\n"
            "This means the classifier could achieve high accuracy by learning content\n"
            "differences rather than genuine AI artifacts.\n\n"
            "This evaluation uses datasets where BOTH classes contain similar content:\n"
            "  - Hemg: 'AiArtData' vs 'RealArt' — both are artwork/art images\n"
            "  - Parveshiiii: balanced binary AI vs Real images\n\n"
            "If our 49 features still achieve high accuracy on these datasets, it provides\n"
            "stronger evidence that the features detect actual AI generation artifacts\n"
            "rather than subject-matter shortcuts."
        )
        ax.text(0.08, 0.64, rationale, transform=ax.transAxes, fontsize=9,
                ha="left", va="top", fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))

        # Summary table
        summary = "Results Summary\n\n"
        for ds in datasets:
            summary += (
                f"Dataset: {ds['dataset']}\n"
                f"  Samples: {ds['n_samples']} ({ds['n_samples']//2} per class)\n"
                f"  XGBoost: {ds['xgb_accuracy']:.1%} acc, {ds['xgb_auc']:.4f} AUC, "
                f"{ds['xgb_precision']:.1%} prec, {ds['xgb_recall']:.1%} rec\n"
                f"  SVM:     {ds['svm_accuracy']:.1%} acc, {ds['svm_auc']:.4f} AUC\n"
                f"  MLP:     {ds['mlp_accuracy']:.1%} acc, {ds['mlp_auc']:.4f} AUC\n\n"
            )
        ax.text(0.08, 0.28, summary, transform=ax.transAxes, fontsize=9,
                ha="left", va="top", fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#66BB6A"))

        pdf.savefig(fig)
        plt.close(fig)

        # ===== PAGE 2+: Per-dataset details =====
        for ds in datasets:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Dataset: {ds['dataset']}", fontsize=14,
                         fontweight="bold", fontfamily="serif", y=0.96)

            # Fold results table
            ax_table = fig.add_axes([0.1, 0.68, 0.8, 0.22])
            ax_table.axis("off")

            if "xgb_folds" in ds:
                table_data = []
                for r in ds["xgb_folds"]:
                    table_data.append([
                        f"Fold {r['fold']}", f"{r['accuracy']:.2%}",
                        f"{r['precision']:.2%}", f"{r['recall']:.2%}",
                        f"{r['f1']:.2%}", f"{r['roc_auc']:.4f}"
                    ])

                accs = [r["accuracy"] for r in ds["xgb_folds"]]
                table_data.append([
                    "Mean +/- Std",
                    f"{np.mean(accs):.2%} +/- {np.std(accs):.2%}",
                    "-", "-", "-",
                    f"{np.mean([r['roc_auc'] for r in ds['xgb_folds']]):.4f}"
                ])

                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Fold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                    loc="center", cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.4)
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor("#4472C4")
                        cell.set_text_props(color="white", fontweight="bold")
                    elif row == len(table_data):
                        cell.set_facecolor("#D6E4F0")

            # Comparison bar chart: XGBoost vs SVM vs MLP
            ax_bar = fig.add_axes([0.1, 0.35, 0.8, 0.25])
            models = ["XGBoost", "SVM", "MLP"]
            accs = [ds["xgb_accuracy"], ds["svm_accuracy"], ds["mlp_accuracy"]]
            aucs = [ds["xgb_auc"], ds["svm_auc"], ds["mlp_auc"]]

            x = np.arange(len(models))
            w = 0.35
            bars1 = ax_bar.bar(x - w/2, accs, w, label="Accuracy", color="#4472C4")
            bars2 = ax_bar.bar(x + w/2, aucs, w, label="ROC-AUC", color="#ED7D31")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(models)
            ax_bar.set_ylim(0, 1.1)
            ax_bar.set_ylabel("Score")
            ax_bar.set_title("Model Comparison", fontsize=11, fontfamily="serif")
            ax_bar.legend()
            ax_bar.grid(axis="y", alpha=0.3)

            for bar in bars1:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{bar.get_height():.1%}", ha="center", fontsize=8)
            for bar in bars2:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{bar.get_height():.3f}", ha="center", fontsize=8)

            # Analysis text
            ax_text = fig.add_axes([0.08, 0.05, 0.84, 0.25])
            ax_text.axis("off")

            best_acc = max(accs)
            best_model = models[accs.index(best_acc)]

            analysis = (
                f"Analysis\n\n"
                f"Dataset: {ds['repo']}\n"
                f"Sample size: {ds['n_samples']} images, {ds['n_features']} features\n\n"
                f"Best model: {best_model} at {best_acc:.1%} accuracy\n\n"
            )
            if best_acc >= 0.80:
                analysis += (
                    "The features demonstrate strong discriminative power even when both\n"
                    "classes contain similar content. This suggests the 49 features capture\n"
                    "genuine AI generation artifacts rather than content-based shortcuts."
                )
            elif best_acc >= 0.65:
                analysis += (
                    "Moderate discriminative power. The features capture some genuine AI\n"
                    "artifacts but performance degrades compared to content-separated datasets,\n"
                    "suggesting prior benchmarks partially relied on content differences."
                )
            else:
                analysis += (
                    "Weak discriminative power on this dataset. The features struggle when\n"
                    "content is controlled, indicating prior high accuracy was largely driven\n"
                    "by subject-matter differences rather than AI detection capability."
                )

            ax_text.text(0, 1, analysis, transform=ax_text.transAxes, fontsize=9,
                        ha="left", va="top", fontfamily="serif")

            pdf.savefig(fig)
            plt.close(fig)

        # ===== FINAL PAGE: Conclusions =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.92, "Conclusions", fontsize=16, fontweight="bold",
                ha="center", va="top", fontfamily="serif", transform=ax.transAxes)

        all_accs = [ds["xgb_accuracy"] for ds in datasets]
        mean_fair_acc = np.mean(all_accs)

        conclusions = (
            f"Mean XGBoost accuracy across fair datasets: {mean_fair_acc:.1%}\n\n"
            "Comparison with previous (potentially confounded) benchmarks:\n"
            "  - Cats vs Bananas (unfair): ~91% accuracy\n"
            "  - WikiArt vs Generated (partially fair): ~92% accuracy\n"
            f"  - Fair evaluation (this report): {mean_fair_acc:.1%} accuracy\n\n"
        )

        if mean_fair_acc >= 0.80:
            conclusions += (
                "CONCLUSION: The 49-feature pipeline holds up under fair evaluation.\n"
                "The accuracy drop from unfair to fair benchmarks is modest, indicating\n"
                "that the features genuinely detect AI artifacts, not just content.\n\n"
                "The frequency-domain features (FFT/DCT) and texture features (GLCM/LBP)\n"
                "appear to be capturing real structural differences between AI-generated\n"
                "and human-created artwork."
            )
        elif mean_fair_acc >= 0.65:
            conclusions += (
                "CONCLUSION: Mixed results. The features have some genuine detection\n"
                "capability but a significant portion of previous accuracy was from\n"
                "content shortcuts. The pipeline needs improvement — likely deeper\n"
                "learned features (self-supervised or fine-tuned ViT) rather than\n"
                "hand-crafted statistics."
            )
        else:
            conclusions += (
                "CONCLUSION: The 49-feature pipeline does NOT generalize to fair\n"
                "evaluation. Previous high accuracy was primarily from content confounds.\n"
                "A fundamentally different approach is needed — likely self-supervised\n"
                "learning of camera/generation-intrinsic features as described in\n"
                "Zhong et al. (2026)."
            )

        conclusions += (
            "\n\nMethodological Note\n\n"
            "This report uses 5-fold stratified cross-validation with 200 images per\n"
            "class. While larger samples would give tighter confidence intervals, this\n"
            "is sufficient to distinguish between >80% and chance-level performance.\n\n"
            "Features: 49 total (39 from Li & Stamp 2025 + 10 FFT/DCT frequency features)\n"
            "Classifiers: XGBoost, SVM (RBF kernel), MLP (100 hidden units)\n"
            "All processing: CPU-only, no pretrained neural networks"
        )

        ax.text(0.08, 0.85, conclusions, transform=ax.transAxes, fontsize=9.5,
                ha="left", va="top", fontfamily="serif")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    results_path = RESULTS_DIR / "fair_evaluation_results.json"
    if not results_path.exists():
        print(f"Run test_fair_evaluation.py first to generate {results_path}")
        sys.exit(1)
    generate_pdf(results_path)
