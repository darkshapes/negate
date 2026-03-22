# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""End-to-end accuracy benchmark using the 39-feature artwork extraction
pipeline from Li & Stamp (2025) "Detecting AI-generated Artwork".

Downloads human artwork from WikiArt and AI-generated images, extracts
39 features (brightness, color, texture, shape, noise), trains SVM/MLP/XGBoost,
reports accuracy with 5-fold cross-validation.

Run with: uv run pytest tests/test_artwork_accuracy.py -v -s
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from datasets import load_dataset, Image as HFImage
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from negate.extract.feature_artwork import ArtworkExtract

# Datasets
HUMAN_ART_REPO = "huggan/wikiart"  # Human artwork (has style labels)
SYNTHETIC_REPO = "exdysa/nano-banana-pro-generated-1k-clone"  # AI-generated
SAMPLE_SIZE = 100  # per class
N_FOLDS = 5
SEED = 42


@pytest.fixture(scope="module")
def benchmark_data():
    """Download images and extract 39 features for both classes."""
    print(f"\nDownloading {SAMPLE_SIZE} human art + {SAMPLE_SIZE} AI images...")

    # Human artwork from WikiArt
    human_ds = load_dataset(HUMAN_ART_REPO, split=f"train[:{SAMPLE_SIZE}]")
    human_ds = human_ds.cast_column("image", HFImage(decode=True, mode="RGB"))

    # AI-generated images
    ai_ds = load_dataset(SYNTHETIC_REPO, split=f"train[:{SAMPLE_SIZE}]")
    ai_ds = ai_ds.cast_column("image", HFImage(decode=True, mode="RGB"))

    extractor = ArtworkExtract()
    features, labels = [], []

    print("Extracting features from human artwork...")
    for row in tqdm(human_ds, total=len(human_ds), desc="Human art"):
        try:
            feat = extractor(row["image"])
            features.append(feat)
            labels.append(0)  # genuine
        except Exception as exc:
            print(f"  Skip: {exc}")

    print("Extracting features from AI images...")
    for row in tqdm(ai_ds, total=len(ai_ds), desc="AI art"):
        try:
            feat = extractor(row["image"])
            features.append(feat)
            labels.append(1)  # synthetic
        except Exception as exc:
            print(f"  Skip: {exc}")

    df = pd.DataFrame(features).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0)
    y = np.array(labels)

    return {
        "X": X, "y": y,
        "feature_names": list(df.columns),
        "n_human": int(np.sum(y == 0)),
        "n_ai": int(np.sum(y == 1)),
    }


@pytest.mark.slow
class TestArtworkDetection:
    """Benchmark the paper's 39-feature approach on artwork detection."""

    def test_feature_extraction(self, benchmark_data):
        """Verify features extracted from both classes."""
        print(f"\n--- Dataset ---")
        print(f"Human art:  {benchmark_data['n_human']}")
        print(f"AI art:     {benchmark_data['n_ai']}")
        print(f"Features:   {benchmark_data['X'].shape[1]}")
        assert benchmark_data["n_human"] >= 50
        assert benchmark_data["n_ai"] >= 50
        assert benchmark_data["X"].shape[1] == 49

    def test_svm_cross_validation(self, benchmark_data):
        """SVM with RBF kernel — paper's best binary model (97.9% reported)."""
        X, y = benchmark_data["X"], benchmark_data["y"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        svm = SVC(C=10, gamma="scale", kernel="rbf", random_state=SEED, probability=True)
        scores = cross_val_score(svm, X_scaled, y, cv=N_FOLDS, scoring="accuracy")

        print(f"\n--- SVM (RBF) {N_FOLDS}-Fold CV ---")
        for i, s in enumerate(scores):
            print(f"  Fold {i+1}: {s:.2%}")
        print(f"  Mean:   {scores.mean():.2%} +/- {scores.std():.2%}")
        print(f"  Paper reports: 97.9% (SVM binary)")

    def test_mlp_cross_validation(self, benchmark_data):
        """MLP — paper's best multiclass model (82% reported)."""
        X, y = benchmark_data["X"], benchmark_data["y"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mlp = MLPClassifier(
            hidden_layer_sizes=(100,), activation="relu", alpha=0.0001,
            solver="adam", max_iter=1000, random_state=SEED,
        )
        scores = cross_val_score(mlp, X_scaled, y, cv=N_FOLDS, scoring="accuracy")

        print(f"\n--- MLP {N_FOLDS}-Fold CV ---")
        for i, s in enumerate(scores):
            print(f"  Fold {i+1}: {s:.2%}")
        print(f"  Mean:   {scores.mean():.2%} +/- {scores.std():.2%}")
        print(f"  Paper reports: 97.6% (MLP binary)")

    def test_xgboost_cross_validation(self, benchmark_data):
        """XGBoost — negate's existing classifier, now with paper's features."""
        X, y = benchmark_data["X"], benchmark_data["y"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_accs, fold_aucs, fold_prec, fold_rec = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": SEED,
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, num_boost_round=200,
                              evals=[(dtest, "test")], early_stopping_rounds=10,
                              verbose_eval=False)

            y_prob = model.predict(dtest)
            y_pred = (y_prob > 0.5).astype(int)
            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_aucs.append(roc_auc_score(y_test, y_prob))
            fold_prec.append(precision_score(y_test, y_pred, zero_division=0))
            fold_rec.append(recall_score(y_test, y_pred, zero_division=0))

        print(f"\n--- XGBoost {N_FOLDS}-Fold CV ---")
        for i, (acc, auc, p, r) in enumerate(zip(fold_accs, fold_aucs, fold_prec, fold_rec)):
            print(f"  Fold {i+1}: acc={acc:.2%} prec={p:.2%} rec={r:.2%} auc={auc:.4f}")
        print(f"  Mean:   acc={np.mean(fold_accs):.2%} prec={np.mean(fold_prec):.2%} rec={np.mean(fold_rec):.2%} auc={np.mean(fold_aucs):.4f}")

    def test_comparison_summary(self, benchmark_data):
        """Print comparison table of all models with precision and recall."""
        X, y = benchmark_data["X"], benchmark_data["y"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        model_results = {}
        for name, clf in [
            ("SVM (RBF)", SVC(C=10, gamma="scale", kernel="rbf", random_state=SEED)),
            ("MLP", MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=1000, random_state=SEED)),
        ]:
            accs, precs, recs = [], [], []
            for train_idx, test_idx in skf.split(X_scaled, y):
                clf_copy = type(clf)(**clf.get_params())
                clf_copy.fit(X_scaled[train_idx], y[train_idx])
                y_pred = clf_copy.predict(X_scaled[test_idx])
                accs.append(accuracy_score(y[test_idx], y_pred))
                precs.append(precision_score(y[test_idx], y_pred, zero_division=0))
                recs.append(recall_score(y[test_idx], y_pred, zero_division=0))
            model_results[name] = {
                "acc": np.array(accs), "prec": np.array(precs), "rec": np.array(recs)
            }

        print(f"\n{'='*75}")
        print(f"  ARTWORK DETECTION: MODEL COMPARISON")
        print(f"  39 features (Li & Stamp 2025) | {len(y)} images")
        print(f"{'='*75}")
        print(f"  {'Model':<15} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'Paper Acc':>11}")
        print(f"  {'-'*57}")
        for name, r in model_results.items():
            paper = {"SVM (RBF)": "97.9%", "MLP": "97.6%"}.get(name, "")
            print(f"  {name:<15} {r['acc'].mean():>9.2%} {r['prec'].mean():>10.2%} {r['rec'].mean():>9.2%} {paper:>11}")
        print(f"  {'Existing negate':<15} {'63.3%':>10} {'--':>11} {'--':>10} {'63.3%':>11}")
        print(f"{'='*75}")
        print(f"\n  Precision = of images flagged as AI, how many actually are (false positive rate)")
        print(f"  Recall    = of actual AI images, how many were caught (false negative rate)")
