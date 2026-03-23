# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
"""Fair evaluation: test artwork features on datasets where both classes are art.

Addresses the confound that previous benchmarks used different subject matter
(cats vs bananas, wikiart vs generated), which inflates accuracy.

Datasets:
  1. Hemg/AI-Generated-vs-Real-Images-Datasets — 153K, "AiArtData" vs "RealArt"
  2. Parveshiiii/AI-vs-Real — 14K balanced binary

We sample N images from each class, extract 49 features, run 5-fold CV,
and report accuracy/precision/recall/F1/AUC with confidence intervals.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from datasets import load_dataset, Image as HFImage
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
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
RESULTS_DIR = Path(__file__).parent.parent / "results"


def extract_all_features(dataset, label_col: str, n_samples: int = 200):
    """Extract features from a dataset, balanced per class."""
    extractor = ArtworkExtract()
    features, labels, errors = [], [], 0

    # Get unique labels and sample equally
    all_labels = dataset[label_col]
    unique_labels = sorted(set(all_labels))
    print(f"  Labels found: {unique_labels}")

    per_class = n_samples // len(unique_labels)

    for lbl in unique_labels:
        indices = [i for i, l in enumerate(all_labels) if l == lbl]
        rng = np.random.RandomState(SEED)
        chosen = rng.choice(indices, size=min(per_class, len(indices)), replace=False)

        for idx in tqdm(chosen, desc=f"  Class {lbl}"):
            try:
                img = dataset[int(idx)]["image"]
                if img is None:
                    errors += 1
                    continue
                if not isinstance(img, Image.Image):
                    errors += 1
                    continue
                feat = extractor(img)
                features.append(feat)
                # Binary: 0 = real/genuine, 1 = AI/synthetic
                labels.append(0 if lbl == max(unique_labels) else 1)
            except Exception as e:
                errors += 1

    print(f"  Extracted {len(features)} images ({errors} errors)")

    df = pd.DataFrame(features).fillna(0)
    X = df.to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0)
    y = np.array(labels)

    return X, y, list(df.columns)


def cross_validate_xgb(X, y):
    """5-fold CV with XGBoost."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr"],
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": spw,
            "seed": SEED,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dtest, "test")], early_stopping_rounds=10,
                          verbose_eval=False)

        y_prob = model.predict(dtest)
        y_pred = (y_prob > 0.5).astype(int)

        fold_results.append({
            "fold": fold + 1,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        })
        all_true.extend(y_test)
        all_prob.extend(y_prob)

    return fold_results, np.array(all_true), np.array(all_prob)


def cross_validate_svm(X, y):
    """5-fold CV with SVM."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []

    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        svm = SVC(kernel="rbf", probability=True, random_state=SEED)
        svm.fit(X_train, y_train)
        y_prob = svm.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_prob.extend(y_prob)

    return np.array(all_true), np.array(all_prob)


def cross_validate_mlp(X, y):
    """5-fold CV with MLP."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_prob = [], []
    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=SEED)
        mlp.fit(X_train, y_train)
        y_prob = mlp.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_prob.extend(y_prob)

    return np.array(all_true), np.array(all_prob)


def summarize(name, fold_results, y_true, y_prob):
    """Print summary for a classifier."""
    y_pred = (y_prob > 0.5).astype(int)
    accs = [r["accuracy"] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: acc={r['accuracy']:.2%} prec={r['precision']:.2%} "
              f"rec={r['recall']:.2%} auc={r['roc_auc']:.4f}")
    print(f"  ---")
    print(f"  Mean acc:  {np.mean(accs):.2%} +/- {np.std(accs):.2%}")
    print(f"  Pooled:    acc={accuracy_score(y_true, y_pred):.2%} "
          f"prec={precision_score(y_true, y_pred, zero_division=0):.2%} "
          f"rec={recall_score(y_true, y_pred, zero_division=0):.2%} "
          f"auc={roc_auc_score(y_true, y_prob):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


def evaluate_dataset(name: str, repo: str, label_col: str, n_samples: int, split: str = "train"):
    """Full evaluation pipeline for one dataset."""
    print(f"\n{'#'*60}")
    print(f"  DATASET: {name}")
    print(f"  Repo: {repo}")
    print(f"  Sampling: {n_samples} images ({n_samples//2} per class)")
    print(f"{'#'*60}")

    print(f"\nLoading dataset...")
    ds = load_dataset(repo, split=split)
    ds = ds.cast_column("image", HFImage(decode=True, mode="RGB"))
    print(f"  Total rows: {len(ds)}")

    X, y, feature_names = extract_all_features(ds, label_col, n_samples)
    print(f"  Class balance: {np.sum(y==0)} real, {np.sum(y==1)} synthetic")
    print(f"  Features: {X.shape[1]}")

    # XGBoost
    print(f"\nRunning {N_FOLDS}-fold CV (XGBoost)...")
    xgb_folds, xgb_true, xgb_prob = cross_validate_xgb(X, y)
    summarize(f"XGBoost on {name}", xgb_folds, xgb_true, xgb_prob)

    # SVM
    print(f"\nRunning {N_FOLDS}-fold CV (SVM)...")
    svm_true, svm_prob = cross_validate_svm(X, y)
    svm_pred = (svm_prob > 0.5).astype(int)
    print(f"  SVM pooled: acc={accuracy_score(svm_true, svm_pred):.2%} "
          f"auc={roc_auc_score(svm_true, svm_prob):.4f}")

    # MLP
    print(f"\nRunning {N_FOLDS}-fold CV (MLP)...")
    mlp_true, mlp_prob = cross_validate_mlp(X, y)
    mlp_pred = (mlp_prob > 0.5).astype(int)
    print(f"  MLP pooled: acc={accuracy_score(mlp_true, mlp_pred):.2%} "
          f"auc={roc_auc_score(mlp_true, mlp_prob):.4f}")

    return {
        "dataset": name,
        "repo": repo,
        "n_samples": int(np.sum(y >= 0)),
        "n_features": X.shape[1],
        "xgb_folds": xgb_folds,
        "xgb_accuracy": float(accuracy_score(xgb_true, (xgb_prob > 0.5).astype(int))),
        "xgb_auc": float(roc_auc_score(xgb_true, xgb_prob)),
        "xgb_precision": float(precision_score(xgb_true, (xgb_prob > 0.5).astype(int), zero_division=0)),
        "xgb_recall": float(recall_score(xgb_true, (xgb_prob > 0.5).astype(int), zero_division=0)),
        "svm_accuracy": float(accuracy_score(svm_true, svm_pred)),
        "svm_auc": float(roc_auc_score(svm_true, svm_prob)),
        "mlp_accuracy": float(accuracy_score(mlp_true, mlp_pred)),
        "mlp_auc": float(roc_auc_score(mlp_true, mlp_prob)),
        "feature_names": feature_names,
    }


def main():
    print("=" * 60)
    print("  FAIR EVALUATION: 49-Feature Artwork Detection")
    print("  Testing on semantically-similar datasets")
    print("=" * 60)

    results = []

    # Dataset 1: Hemg — both classes are art
    results.append(evaluate_dataset(
        name="AI-Art vs Real-Art (Hemg)",
        repo="Hemg/AI-Generated-vs-Real-Images-Datasets",
        label_col="label",
        n_samples=400,
    ))

    # Dataset 2: Parveshiiii — balanced binary
    results.append(evaluate_dataset(
        name="AI vs Real (Parveshiiii)",
        repo="Parveshiiii/AI-vs-Real",
        label_col="binary_label",
        n_samples=400,
    ))

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "fair_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "evaluation": "fair_semantically_similar",
            "datasets": results,
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['dataset']}:")
        print(f"    XGBoost: acc={r['xgb_accuracy']:.2%} auc={r['xgb_auc']:.4f} "
              f"prec={r['xgb_precision']:.2%} rec={r['xgb_recall']:.2%}")
        print(f"    SVM:     acc={r['svm_accuracy']:.2%} auc={r['svm_auc']:.4f}")
        print(f"    MLP:     acc={r['mlp_accuracy']:.2%} auc={r['mlp_auc']:.4f}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
