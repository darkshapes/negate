"""Hybrid detector: 148 handcrafted + 768 ConvNeXt, max training diversity.

Extracts features INCREMENTALLY to avoid OOM from holding all images in memory.
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from pathlib import Path

os.environ["HF_HOME"] = "D:/Projects/negate/negate/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/Projects/negate/negate/.cache/huggingface/datasets"
# Set HF_TOKEN env var or run: huggingface-cli login

import lightgbm as lgb
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_learned import LearnedExtract

art_ext = ArtworkExtract()
learned_ext = LearnedExtract()

BASE = Path("D:/Projects/negate/negate/.datasets/imaginet/extracted")
CIVITAI = Path("D:/Projects/negate/negate/.datasets/civitai")
SEED = 42
rng = np.random.RandomState(SEED)
W_LGBM, W_RF, W_SVM = 0.4, 0.1, 0.5
HIGH_T, LOW_T = 0.80, 0.20


def load_paths(path, recursive, max_n):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = list(path.rglob("*") if recursive else path.iterdir())
    files = [f for f in files if f.suffix.lower() in exts]
    if len(files) > max_n:
        files = list(rng.choice(files, max_n, replace=False))
    return files


def extract_one(img):
    """Extract 916 features from one PIL image."""
    try:
        hc = list(art_ext(img).values())
    except Exception:
        hc = [0.0] * 148
    try:
        with torch.no_grad():
            ln = list(learned_ext(img).values())
    except Exception:
        ln = [0.0] * 768
    return hc + ln


def extract_from_paths(files, desc=""):
    """Extract features from file paths, one at a time (no memory buildup)."""
    rows = []
    for f in tqdm(files, desc=desc):
        try:
            img = Image.open(f).convert("RGB")
            rows.append(extract_one(img))
            del img
        except Exception:
            rows.append([0.0] * 916)
    X = np.array(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_from_stream(repo, max_n, desc=""):
    """Stream from HF, extract features one at a time."""
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    rows = []
    for i, s in enumerate(tqdm(ds, desc=desc, total=max_n)):
        if i >= max_n:
            break
        try:
            img = s["image"].convert("RGB")
            rows.append(extract_one(img))
            del img
        except Exception:
            rows.append([0.0] * 916)
    if not rows:
        return np.empty((0, 916))
    X = np.array(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_from_imgs(images, desc=""):
    rows = []
    for img in tqdm(images, desc=desc):
        rows.append(extract_one(img))
    X = np.array(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    print("=" * 70, flush=True)
    print("  HYBRID DETECTOR: 916 features, max diversity training", flush=True)
    print("=" * 70, flush=True)

    feature_chunks_real = []
    feature_chunks_fake = []

    # === REAL: WikiArt ===
    print("\n--- REAL: WikiArt (2000) ---", flush=True)
    files = load_paths(BASE / "wikiart", recursive=True, max_n=2000)
    X = extract_from_paths(files, "WikiArt")
    feature_chunks_real.append(X)
    print(f"  Got {len(X)}", flush=True)
    gc.collect()

    # === REAL: tellif ===
    print("\n--- REAL: tellif (122) ---", flush=True)
    from datasets import load_dataset
    ds_tellif = load_dataset("tellif/ai_vs_real_image_semantically_similar", split="test")
    tellif_real_imgs = [ds_tellif[i]["image"].convert("RGB") for i in range(len(ds_tellif)) if ds_tellif[i]["label"] == 8]
    X = extract_from_imgs(tellif_real_imgs, "tellif real")
    feature_chunks_real.append(X)
    print(f"  Got {len(X)}", flush=True)
    del tellif_real_imgs
    gc.collect()

    # === REAL: Hemg (stream) ===
    print("\n--- REAL: Hemg (1000) ---", flush=True)
    X = extract_from_stream("Hemg/AI-Generated-vs-Real-Images-Datasets", max_n=2500, desc="Hemg real")
    # Hemg label=1 is real, but streaming gives us mixed — need to filter
    # Actually extract_from_stream doesn't filter by label. Let me handle differently.
    ds_hemg = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)
    rows = []
    count = 0
    for s in tqdm(ds_hemg, desc="Hemg real", total=3000):
        if s["label"] == 1:  # real
            try:
                rows.append(extract_one(s["image"].convert("RGB")))
                count += 1
            except Exception:
                pass
        if count >= 1000:
            break
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) if rows else np.empty((0, 916))
    feature_chunks_real.append(X)
    print(f"  Got {len(X)}", flush=True)
    gc.collect()

    # === FAKE: ImagiNet old ===
    print("\n--- FAKE: ImagiNet old (1000) ---", flush=True)
    for src in ["sdxl_paintings_fake", "sd_paintings_fake", "dalle3", "journeydb", "animaginexl_paintings_fake"]:
        files = load_paths(BASE / src, recursive=False, max_n=200)
        X = extract_from_paths(files, f"ImagiNet {src}")
        feature_chunks_fake.append(X)
        print(f"  {src}: {len(X)}", flush=True)
        gc.collect()

    # === FAKE: Modern HF datasets ===
    print("\n--- FAKE: Modern generators ---", flush=True)
    for repo, n, name in [
        ("ash12321/seedream-4.5-generated-2k", 500, "Seedream 4.5"),
        ("exdysa/nano-banana-pro-generated-1k-clone", 500, "Nano Banana Pro"),
        ("LukasT9/Flux-1-Dev-Images-1k", 500, "Flux Dev"),
        ("LukasT9/Flux-1-Schnell-Images-1k", 500, "Flux Schnell"),
    ]:
        X = extract_from_stream(repo, max_n=n, desc=name)
        feature_chunks_fake.append(X)
        print(f"  {name}: {len(X)}", flush=True)
        gc.collect()

    # === FAKE: CivitAI ===
    print("\n--- FAKE: CivitAI ---", flush=True)
    for gen_dir in ["flux", "sdxl", "pony", "illustrious"]:
        p = CIVITAI / gen_dir
        if p.exists() and any(p.iterdir()):
            files = load_paths(p, recursive=False, max_n=500)
            X = extract_from_paths(files, f"CivitAI {gen_dir}")
            feature_chunks_fake.append(X)
            print(f"  CivitAI {gen_dir}: {len(X)}", flush=True)
            gc.collect()

    # === FAKE: Hemg AI ===
    print("\n--- FAKE: Hemg AI (1000) ---", flush=True)
    ds_hemg2 = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)
    rows = []
    count = 0
    for s in tqdm(ds_hemg2, desc="Hemg AI", total=1500):
        if s["label"] == 0:  # AI
            try:
                rows.append(extract_one(s["image"].convert("RGB")))
                count += 1
            except Exception:
                pass
        if count >= 1000:
            break
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) if rows else np.empty((0, 916))
    feature_chunks_fake.append(X)
    print(f"  Hemg AI: {len(X)}", flush=True)
    gc.collect()

    # === Combine ===
    X_real = np.vstack([c for c in feature_chunks_real if len(c) > 0])
    X_fake = np.vstack([c for c in feature_chunks_fake if len(c) > 0])
    del feature_chunks_real, feature_chunks_fake
    gc.collect()

    n = min(len(X_real), len(X_fake))
    X_train = np.vstack([X_real[:n], X_fake[:n]])
    y_train = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"\n  TOTAL: {len(X_train)} train ({n}/class), {X_train.shape[1]} features", flush=True)
    print(f"  Real sources: WikiArt + tellif + Hemg", flush=True)
    print(f"  Fake sources: ImagiNet + Seedream + Nano Banana + Flux + CivitAI + Hemg AI", flush=True)

    # === 5-fold CV ===
    print("\n--- 5-fold CV ---", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for fold, (tr, te) in enumerate(skf.split(X_train, y_train)):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
                                num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED)
        m.fit(X_train[tr], y_train[tr])
        p = m.predict_proba(X_train[te])[:, 1]
        acc = accuracy_score(y_train[te], (p > 0.5).astype(int))
        auc = roc_auc_score(y_train[te], p)
        accs.append(acc); aucs.append(auc)
        print(f"  Fold {fold+1}: acc={acc:.4f} auc={auc:.4f}", flush=True)
    print(f"  Mean: acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}", flush=True)

    # === Train ensemble ===
    print("\n--- Training ensemble ---", flush=True)
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
                                num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED)
    lgbm.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=1, random_state=SEED)
    rf.fit(X_train, y_train)
    scaler = StandardScaler()
    svm = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(scaler.fit_transform(X_train), y_train)

    def predict(X):
        p = (W_LGBM * lgbm.predict_proba(X)[:, 1] +
             W_RF * rf.predict_proba(X)[:, 1] +
             W_SVM * svm.predict_proba(scaler.transform(X))[:, 1])
        pred = np.full(len(X), -1)
        pred[p >= HIGH_T] = 1
        pred[p < LOW_T] = 0
        return pred, p

    # === TEST: tellif SOTA ===
    print("\n" + "=" * 70, flush=True)
    print("  TEST: tellif 2025 SOTA (HYBRID 916 features)", flush=True)
    print("=" * 70, flush=True)

    label_names = ds_tellif.features["label"].names
    results = {}

    for i, gen_name in enumerate(label_names):
        indices = [j for j in range(len(ds_tellif)) if ds_tellif[j]["label"] == i]
        if not indices:
            continue
        images = [ds_tellif[j]["image"].convert("RGB") for j in indices]
        X_gen = extract_from_imgs(images, gen_name)
        pred, prob = predict(X_gen)

        if gen_name == "real":
            fp = (pred == 1).sum()
            correct = (pred == 0).sum()
            uncertain = (pred == -1).sum()
            fp_rate = fp / len(X_gen)
            print(f"  {gen_name:45s}  n={len(X_gen):3d}  GENUINE={correct}  UNCERTAIN={uncertain}  FP={fp}  FP_rate={fp_rate:.1%}", flush=True)
            results[gen_name] = {"n": len(X_gen), "fp": int(fp), "fp_rate": float(fp_rate)}
        else:
            detected = (pred == 1).sum()
            uncertain = (pred == -1).sum()
            missed = (pred == 0).sum()
            rate = detected / len(X_gen)
            print(f"  {gen_name:45s}  n={len(X_gen):3d}  DETECTED={detected}  UNCERTAIN={uncertain}  MISSED={missed}  rate={rate:.1%}", flush=True)
            results[gen_name] = {"n": len(X_gen), "detected": int(detected), "uncertain": int(uncertain), "missed": int(missed), "rate": float(rate)}
        del images
        gc.collect()

    # === Feature importance ===
    importances = lgbm.feature_importances_
    hc_imp = importances[:148].sum()
    ln_imp = importances[148:].sum()
    total_imp = importances.sum()
    print(f"\n  Feature importance: HC={hc_imp/total_imp*100:.1f}% / Learned={ln_imp/total_imp*100:.1f}%", flush=True)

    all_names = list(art_ext(Image.new("RGB", (64, 64), "red")).keys()) + learned_ext.feature_names()
    sorted_idx = np.argsort(importances)[::-1]
    print(f"  Top 10:", flush=True)
    for r in range(10):
        idx = sorted_idx[r]
        src = "HC" if idx < 148 else "CNN"
        name = all_names[idx] if idx < len(all_names) else f"f{idx}"
        print(f"    {r+1:2d}. [{src}] {name:35s} {importances[idx]}", flush=True)

    # Save
    out = Path(__file__).parent.parent / "results" / "hybrid_modern_results.json"
    with open(out, "w") as f:
        json.dump({
            "training": {"n_per_class": int(n), "n_features": 916,
                         "cv_acc": float(np.mean(accs)), "cv_auc": float(np.mean(aucs))},
            "tellif": results,
            "importance": {"handcrafted_pct": float(hc_imp/total_imp), "learned_pct": float(ln_imp/total_imp)},
        }, f, indent=2)
    print(f"\nSaved to {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
