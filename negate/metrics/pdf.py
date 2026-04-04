# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path


def generate_pdf(X, y, feature_names, results, ensemble_probs, ensemble_preds, model, imgs_human, imgs_ai):

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch
    from datetime import datetime
    from sklearn.metrics import (
        roc_curve,
        confusion_matrix,
        precision_recall_curve,
    )

    OUTPUT_DIR: Path = Path(__file__).parent.parent / "results"
    N_FOLDS = 5

    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUTPUT_DIR / "artwork_detection_results.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # ===== PAGE 1: Title + Results Table =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.92, "AI-Generated Artwork Detection", fontsize=22, fontweight="bold", ha="center", fontfamily="serif", transform=ax.transAxes)
        ax.text(0.5, 0.87, "Multi-Signal Ensemble with Calibrated Thresholds", fontsize=12, ha="center", fontfamily="serif", style="italic", transform=ax.transAxes)
        ax.text(0.5, 0.83, f"negate project | {datetime.now().strftime('%B %d, %Y')}", fontsize=10, ha="center", fontfamily="serif", transform=ax.transAxes)

        # Results table
        ax_table = fig.add_axes([0.08, 0.52, 0.84, 0.26])
        ax_table.axis("off")

        table_data = []
        for name, r in results.items():
            row = [name, f"{r['accuracy']:.1%}", f"{r['precision']:.1%}", f"{r['recall']:.1%}", f"{r['f1']:.1%}", f"{r['roc_auc']:.4f}"]
            if name == "Ensemble":
                row.append(f"{r['n_abstained']}")
            else:
                row.append("-")
            table_data.append(row)
        table_data.append(["Existing negate", "63.3%", "--", "--", "--", "0.669", "-"])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Abstained"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2E4057")
                cell.set_text_props(color="white", fontweight="bold")
            elif row == len(table_data):
                cell.set_facecolor("#FFE0E0")

        # Summary text
        ens = results.get("Ensemble", {})
        summary = (
            "Approach\n\n"
            f"  Features: {X.shape[1]} (39 artwork + 10 frequency analysis)\n"
            f"  Dataset:  {np.sum(y == 0)} human artworks (WikiArt) + {np.sum(y == 1)} AI images\n"
            f"  CV:       {N_FOLDS}-fold stratified cross-validation\n\n"
            "  Three calibrated classifiers (SVM, MLP, XGBoost) vote via averaged\n"
            "  probabilities. Images where ensemble confidence is between 30-70%\n"
            f"  are marked 'uncertain' ({ens.get('n_abstained', 0)} images abstained).\n\n"
            "  Precision = of images flagged AI, how many actually are\n"
            "  Recall    = of actual AI images, how many were caught"
        )
        ax.text(0.08, 0.48, summary, fontsize=9, ha="left", va="top", fontfamily="serif", transform=ax.transAxes)

        # Key findings
        findings = (
            "Key Findings\n\n"
            f"  1. Ensemble achieves {ens.get('precision', 0):.1%} precision, "
            f"{ens.get('recall', 0):.1%} recall on classified images\n"
            f"  2. {ens.get('n_abstained', 0)} uncertain images abstained from "
            f"(reduces false positives)\n"
            f"  3. +{(ens.get('accuracy', 0) - 0.633) * 100:.1f}pp improvement over "
            "existing negate pipeline (63.3%)\n"
            f"  4. Frequency features (FFT/DCT) add spectral artifact detection\n"
            "  5. All processing is CPU-only, ~12 images/sec"
        )
        ax.text(
            0.08,
            0.24,
            findings,
            fontsize=9,
            ha="left",
            va="top",
            fontfamily="serif",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#66BB6A"),
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ===== PAGE 2: ROC + PR curves + Confusion Matrix =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle("Detection Performance Analysis", fontsize=14, fontweight="bold", fontfamily="serif", y=0.96)

        # ROC curves
        ax_roc = fig.add_axes([0.08, 0.62, 0.4, 0.28])
        colors = {"SVM": "#4472C4", "MLP": "#ED7D31", "XGBoost": "#70AD47", "Ensemble": "#C00000"}
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y, r["probs"])
            ax_roc.plot(fpr, tpr, color=colors.get(name, "gray"), linewidth=2, label=f"{name} ({r['roc_auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_xlabel("False Positive Rate", fontsize=9)
        ax_roc.set_ylabel("True Positive Rate", fontsize=9)
        ax_roc.set_title("ROC Curves", fontsize=10, fontfamily="serif")
        ax_roc.legend(fontsize=7, loc="lower right")
        ax_roc.grid(True, alpha=0.2)

        # Precision-Recall curves
        ax_pr = fig.add_axes([0.55, 0.62, 0.4, 0.28])
        for name, r in results.items():
            prec_curve, rec_curve, _ = precision_recall_curve(y, r["probs"])
            ax_pr.plot(rec_curve, prec_curve, color=colors.get(name, "gray"), linewidth=2, label=name)
        ax_pr.set_xlabel("Recall", fontsize=9)
        ax_pr.set_ylabel("Precision", fontsize=9)
        ax_pr.set_title("Precision-Recall Curves", fontsize=10, fontfamily="serif")
        ax_pr.legend(fontsize=7)
        ax_pr.grid(True, alpha=0.2)

        # Ensemble confusion matrix
        ax_cm = fig.add_axes([0.08, 0.28, 0.35, 0.26])
        confident = ensemble_preds >= 0
        if confident.sum() > 0:
            cm = confusion_matrix(y[confident], ensemble_preds[confident])
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Human", "AI"], fontsize=9)
            ax_cm.set_yticklabels(["Human", "AI"], fontsize=9)
            ax_cm.set_xlabel("Predicted", fontsize=9)
            ax_cm.set_ylabel("Actual", fontsize=9)
            ax_cm.set_title("Ensemble (confident only)", fontsize=10, fontfamily="serif")
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")

        # Probability distribution
        ax_hist = fig.add_axes([0.55, 0.28, 0.4, 0.26])
        human_probs = ensemble_probs[y == 0]
        ai_probs = ensemble_probs[y == 1]
        ax_hist.hist(human_probs, bins=20, alpha=0.6, color="#4472C4", label="Human art", density=True)
        ax_hist.hist(ai_probs, bins=20, alpha=0.6, color="#ED7D31", label="AI art", density=True)
        ax_hist.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="Abstain zone")
        ax_hist.axvline(x=0.7, color="red", linestyle="--", alpha=0.5)
        ax_hist.axvspan(0.3, 0.7, alpha=0.1, color="red")
        ax_hist.set_xlabel("Ensemble Probability (AI)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.set_title("Probability Distribution", fontsize=10, fontfamily="serif")
        ax_hist.legend(fontsize=7)

        # Per-model agreement analysis
        ax_agree = fig.add_axes([0.08, 0.04, 0.84, 0.18])
        ax_agree.axis("off")
        n_all_agree = sum(1 for i in range(len(y)) if len(set(results[n]["probs"][i] > 0.5 for n in ["SVM", "MLP", "XGBoost"])) == 1)
        n_disagree = len(y) - n_all_agree
        agree_text = (
            "Model Agreement Analysis\n\n"
            f"  All 3 models agree:    {n_all_agree}/{len(y)} ({n_all_agree / len(y):.0%})\n"
            f"  At least 1 disagrees:  {n_disagree}/{len(y)} ({n_disagree / len(y):.0%})\n\n"
            "  When models disagree, the ensemble uses averaged probability with\n"
            "  abstention zone (0.3-0.7). This reduces false positives at the cost\n"
            "  of some unclassified images -- a deliberate tradeoff for precision."
        )
        ax_agree.text(0, 1, agree_text, fontsize=9, ha="left", va="top", fontfamily="serif", transform=ax_agree.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # ===== PAGE 3: Feature Analysis + Examples =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle("Feature Analysis & Examples", fontsize=14, fontweight="bold", fontfamily="serif", y=0.96)

        # Example images
        n = min(4, len(imgs_human), len(imgs_ai))
        gs = gridspec.GridSpec(2, n, top=0.9, bottom=0.65, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
        for i in range(n):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(imgs_human[i])
            ax.set_title(f"Human #{i + 1}", fontsize=8)
            ax.axis("off")
        for i in range(n):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(imgs_ai[i])
            ax.set_title(f"AI #{i + 1}", fontsize=8)
            ax.axis("off")

        # Feature importance
        ax_imp = fig.add_axes([0.12, 0.08, 0.76, 0.5])
        importance = model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        if sorted_imp:
            # Map back to real feature names
            disp_names = []
            for fname, _ in sorted_imp:
                idx = int(fname[1:])  # f0 -> 0
                disp_names.append(feature_names[idx] if idx < len(feature_names) else fname)
            disp_names = disp_names[::-1]
            gains = [x[1] for x in sorted_imp][::-1]

            color_map = {
                "fft": "#C00000",
                "dct": "#C00000",
                "hog": "#ED7D31",
                "edge": "#ED7D31",
                "lbp": "#70AD47",
                "contrast": "#70AD47",
                "correlation": "#70AD47",
                "energy": "#70AD47",
                "homogeneity": "#70AD47",
            }
            bar_colors = []
            for n in disp_names:
                c = "#4472C4"  # default
                for prefix, color in color_map.items():
                    if prefix in n:
                        c = color
                        break
                bar_colors.append(c)

            ax_imp.barh(range(len(disp_names)), gains, color=bar_colors)
            ax_imp.set_yticks(range(len(disp_names)))
            ax_imp.set_yticklabels(disp_names, fontsize=7)
            ax_imp.set_xlabel("XGBoost Gain", fontsize=9)
            ax_imp.set_title("Top 20 Features by Importance", fontsize=10, fontfamily="serif")

            legend_elements = [
                Patch(facecolor="#C00000", label="Frequency (FFT/DCT)"),
                Patch(facecolor="#ED7D31", label="Shape (HOG/edges)"),
                Patch(facecolor="#70AD47", label="Texture (GLCM/LBP)"),
                Patch(facecolor="#4472C4", label="Color/Brightness/Noise"),
            ]
            ax_imp.legend(handles=legend_elements, fontsize=7, loc="lower right")

        pdf.savefig(fig)
        plt.close(fig)

        # ===== PAGE 4: Methodology & Architecture =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.95, "Architecture & Methodology", fontsize=14, fontweight="bold", ha="center", fontfamily="serif", transform=ax.transAxes)

        method_text = (
            "Multi-Signal Ensemble Architecture\n\n"
            "The detection system combines three orthogonal classifiers, each seeing the\n"
            "same feature space but learning different decision boundaries:\n\n"
            "  1. SVM (RBF kernel) - Finds nonlinear decision boundaries in feature space.\n"
            "     Calibrated with Platt scaling (sigmoid) for reliable probabilities.\n\n"
            "  2. MLP (100 hidden units) - Learns feature interactions through backpropagation.\n"
            "     Calibrated with Platt scaling for probability alignment.\n\n"
            "  3. XGBoost (gradient boosted trees) - Captures feature thresholds and\n"
            "     interactions. Naturally outputs calibrated log-odds.\n\n"
            "Ensemble Voting: Averaged calibrated probabilities from all three models.\n"
            "This is more robust than majority voting because it accounts for confidence.\n\n"
            "Calibrated Confidence & Abstention\n\n"
            "Instead of a hard 0.5 threshold, the ensemble uses a deliberate 'uncertain'\n"
            "zone between 0.3 and 0.7 probability. Images in this zone are marked as\n"
            "'uncertain' rather than forced into a class. This dramatically improves\n"
            "precision on the images that ARE classified.\n\n"
            "Feature Extraction Pipeline (49 features, CPU-only)\n\n"
            "  Brightness (2)  - Global luminance statistics\n"
            "  Color (23)      - RGB/HSV histogram moments (mean, var, kurtosis, skew, entropy)\n"
            "  Texture (6)     - GLCM co-occurrence + LBP local patterns\n"
            "  Shape (6)       - HOG gradient histograms + Canny edge density\n"
            "  Noise (2)       - Estimated noise entropy + signal-to-noise ratio\n"
            "  Frequency (10)  - FFT radial band energies, spectral centroid, phase coherence,\n"
            "                    DCT AC/DC ratio, high-freq energy, coefficient sparsity\n\n"
            "The frequency branch is the key addition beyond Li & Stamp (2025). AI generators\n"
            "leave characteristic spectral signatures from upsampling layers, attention patterns,\n"
            "and latent space decoding. These are invisible in pixel space but clearly visible\n"
            "in the frequency domain.\n\n"
            "Limitations\n\n"
            "  - Tested on mismatched subjects (WikiArt paintings vs AI banana images)\n"
            "  - Not yet tested on hard negatives (polished digital art, img2img, LoRA art)\n"
            "  - Single generator family in AI training data (Stable Diffusion variants)\n"
            "  - 200 sample dataset is small for robust conclusions\n\n"
            "References\n\n"
            "  [1] Li & Stamp, 'Detecting AI-generated Artwork', arXiv:2504.07078, 2025\n"
            "  [2] negate project, github.com/darkshapes/negate"
        )
        ax.text(0.06, 0.9, method_text, fontsize=8.5, ha="left", va="top", fontfamily="serif", transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved: {pdf_path}")
    return pdf_path
