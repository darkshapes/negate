"""ImagiNet Paintings Benchmark: per-generator accuracy on labeled art data.

Tests the 148-feature extractor against 6 different AI generators using
real WikiArt paintings as the genuine class. This is the definitive test
for art-specific detection with known generators.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract

ext = ArtworkExtract()
BASE = Path("D:/Projects/negate/negate/.datasets/imaginet/extracted")
SEED = 42
MAX_PER_CLASS = 500
rng = np.random.RandomState(SEED)

GENERATORS = {
    "wikiart": {"path": BASE / "wikiart", "label": 0, "recursive": True},
    "AnimagineXL": {"path": BASE / "animaginexl_paintings_fake", "label": 1, "recursive": False},
    "SD": {"path": BASE / "sd_paintings_fake", "label": 1, "recursive": False},
    "SDXL": {"path": BASE / "sdxl_paintings_fake", "label": 1, "recursive": False},
    "StyleGAN": {"path": BASE / "wikiart_stylegan", "label": 1, "recursive": True},
    "Midjourney": {"path": BASE / "journeydb", "label": 1, "recursive": False},
    "DALL-E_3": {"path": BASE / "dalle3", "label": 1, "recursive": False},
}


def load_image_paths(path: Path, recursive: bool, max_n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if recursive:
        files = [f for f in path.rglob("*") if f.suffix.lower() in exts]
    else:
        files = [f for f in path.iterdir() if f.suffix.lower() in exts]
    if len(files) > max_n:
        files = list(rng.choice(files, max_n, replace=False))
    return files


def extract_features(files: list[Path], name: str) -> np.ndarray:
    rows = []
    for f in tqdm(files, desc=name):
        try:
            img = Image.open(f).convert("RGB")
            feat = ext(img)
            rows.append(list(feat.values()))
        except Exception:
            rows.append([0.0] * 148)
    X = np.array(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cv_evaluate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics = {"acc": [], "auc": [], "f1": [], "prec": [], "rec": []}
    for tr, te in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED,
        )
        model.fit(X[tr], y[tr])
        yp = model.predict_proba(X[te])[:, 1]
        ypred = (yp > 0.5).astype(int)
        metrics["acc"].append(accuracy_score(y[te], ypred))
        metrics["auc"].append(roc_auc_score(y[te], yp))
        metrics["f1"].append(f1_score(y[te], ypred, average="macro"))
        metrics["prec"].append(precision_score(y[te], ypred))
        metrics["rec"].append(recall_score(y[te], ypred))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    print("=" * 60)
    print("  ImagiNet PAINTINGS Benchmark")
    print("  148 features, LightGBM, per-generator accuracy")
    print("=" * 60)

    # Load and extract features
    all_features = {}
    for name, info in GENERATORS.items():
        files = load_image_paths(info["path"], info["recursive"], MAX_PER_CLASS)
        print(f"\n{name}: {len(files)} images (label={info['label']})")
        all_features[name] = extract_features(files, name)
        print(f"  Shape: {all_features[name].shape}")

    X_real = all_features["wikiart"]

    # --- EXP 1: Per-generator ---
    print("\n" + "=" * 60)
    print("  EXP 1: Per-Generator (Real Art vs Each Generator)")
    print("=" * 60)

    results_per_gen = {}
    for gen in ["AnimagineXL", "SD", "SDXL", "StyleGAN", "Midjourney", "DALL-E_3"]:
        X_fake = all_features[gen]
        n = min(len(X_real), len(X_fake))
        X = np.vstack([X_real[:n], X_fake[:n]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        r = cv_evaluate(X, y)
        results_per_gen[gen] = r
        print(f"  {gen:15s}  acc={r['acc']:.4f}  auc={r['auc']:.4f}  f1={r['f1']:.4f}  prec={r['prec']:.4f}  rec={r['rec']:.4f}")

    # --- EXP 2: Pooled ---
    print("\n" + "=" * 60)
    print("  EXP 2: Pooled (All Generators Mixed)")
    print("=" * 60)

    fake_list = ["AnimagineXL", "SD", "SDXL", "StyleGAN", "Midjourney", "DALL-E_3"]
    X_all_fake = np.vstack([all_features[g] for g in fake_list])
    n_fake = len(X_all_fake)
    n_real = len(X_real)
    if n_real < n_fake:
        idx = rng.choice(n_fake, n_real, replace=False)
        X_all_fake = X_all_fake[idx]
    else:
        idx = rng.choice(n_real, n_fake, replace=False)
        X_real_pooled = X_real[idx]

    n = min(len(X_real), len(X_all_fake))
    X_pooled = np.vstack([X_real[:n], X_all_fake[:n]])
    y_pooled = np.concatenate([np.zeros(n), np.ones(n)])
    r = cv_evaluate(X_pooled, y_pooled)
    print(f"  Pooled: acc={r['acc']:.4f}  auc={r['auc']:.4f}  f1={r['f1']:.4f}  prec={r['prec']:.4f}  rec={r['rec']:.4f}")

    # --- EXP 3: Leave-one-generator-out ---
    print("\n" + "=" * 60)
    print("  EXP 3: Leave-One-Generator-Out (Generalization)")
    print("=" * 60)

    results_logo = {}
    for held_out in fake_list:
        train_gens = [g for g in fake_list if g != held_out]
        X_train_fake = np.vstack([all_features[g] for g in train_gens])
        n_tf = len(X_train_fake)
        X_train = np.vstack([X_real[:n_tf], X_train_fake])
        y_train = np.concatenate([np.zeros(min(len(X_real), n_tf)), np.ones(n_tf)])

        X_test_fake = all_features[held_out]
        n_te = len(X_test_fake)
        X_test = np.vstack([X_real[:n_te], X_test_fake])
        y_test = np.concatenate([np.zeros(n_te), np.ones(n_te)])

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED,
        )
        model.fit(X_train, y_train)
        yp = model.predict_proba(X_test)[:, 1]
        ypred = (yp > 0.5).astype(int)
        acc = accuracy_score(y_test, ypred)
        auc = roc_auc_score(y_test, yp)
        f1 = f1_score(y_test, ypred, average="macro")
        results_logo[held_out] = {"acc": acc, "auc": auc, "f1": f1}
        print(f"  Hold out {held_out:15s}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}")

    # Save results
    results = {
        "per_generator": results_per_gen,
        "pooled": r,
        "leave_one_out": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results_logo.items()},
    }
    out_path = Path(__file__).parent.parent / "results" / "imaginet_paintings_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
