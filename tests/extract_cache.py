"""Feature extraction with disk caching. Pause/resume safe.

Saves extracted features to .npz files after each data source.
On restart, loads from cache and skips already-extracted sources.

Usage:
    uv run python tests/extract_cache.py          # extract all
    uv run python tests/extract_cache.py --train   # train + test using cached features
"""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["HF_HOME"] = "D:/Projects/negate/negate/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/Projects/negate/negate/.cache/huggingface/datasets"
# Set HF_TOKEN env var or run: huggingface-cli login

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_learned import LearnedExtract

CACHE_DIR = Path("D:/Projects/negate/negate/.cache/features")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path("D:/Projects/negate/negate/.datasets/imaginet/extracted")
CIVITAI = Path("D:/Projects/negate/negate/.datasets/civitai")
SEED = 42


def get_extractors():
    return ArtworkExtract(), LearnedExtract()


def extract_one(art_ext, learned_ext, img):
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


def extract_and_cache(name, image_source, art_ext, learned_ext):
    """Extract features and save to cache. Skip if already cached."""
    cache_file = CACHE_DIR / f"{name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  {name}: loaded from cache ({len(data['X'])} images)", flush=True)
        return data["X"]

    print(f"  {name}: extracting...", flush=True)
    rows = []
    for img in tqdm(image_source, desc=name):
        rows.append(extract_one(art_ext, learned_ext, img))
        del img

    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(cache_file, X=X)
    print(f"  {name}: cached {len(X)} images to {cache_file}", flush=True)
    gc.collect()
    return X


def load_paths_as_images(path, recursive, max_n):
    """Generator that yields PIL images from file paths."""
    rng = np.random.RandomState(SEED)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = list(path.rglob("*") if recursive else path.iterdir())
    files = [f for f in files if f.suffix.lower() in exts]
    if len(files) > max_n:
        files = list(rng.choice(files, max_n, replace=False))
    for f in files:
        try:
            yield Image.open(f).convert("RGB")
        except Exception:
            pass


def stream_hf_images(repo, max_n, label_filter=None):
    """Generator that yields PIL images from HuggingFace streaming."""
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    count = 0
    for s in ds:
        if label_filter is not None and s.get("label") != label_filter:
            continue
        try:
            yield s["image"].convert("RGB")
            count += 1
        except Exception:
            pass
        if count >= max_n:
            break


def extract_all():
    """Extract and cache all data sources."""
    art_ext, learned_ext = get_extractors()

    print("=" * 60, flush=True)
    print("  FEATURE EXTRACTION WITH CACHING", flush=True)
    print("  Pause anytime — restart will skip completed sources", flush=True)
    print("=" * 60, flush=True)

    # === REAL ===
    print("\n--- REAL ---", flush=True)

    extract_and_cache("real_wikiart",
                      load_paths_as_images(BASE / "wikiart", recursive=True, max_n=1500),
                      art_ext, learned_ext)

    from datasets import load_dataset
    ds_tellif = load_dataset("tellif/ai_vs_real_image_semantically_similar", split="test")
    tellif_real = [ds_tellif[i]["image"].convert("RGB") for i in range(len(ds_tellif)) if ds_tellif[i]["label"] == 8]
    extract_and_cache("real_tellif", tellif_real, art_ext, learned_ext)
    del tellif_real; gc.collect()

    extract_and_cache("real_hemg",
                      stream_hf_images("Hemg/AI-Generated-vs-Real-Images-Datasets", max_n=800, label_filter=1),
                      art_ext, learned_ext)

    # === FAKE: old generators ===
    print("\n--- FAKE (old) ---", flush=True)
    for src in ["sdxl_paintings_fake", "sd_paintings_fake", "dalle3", "journeydb", "animaginexl_paintings_fake"]:
        extract_and_cache(f"fake_imaginet_{src}",
                          load_paths_as_images(BASE / src, recursive=False, max_n=100),
                          art_ext, learned_ext)

    # === FAKE: modern HF datasets ===
    print("\n--- FAKE (modern) ---", flush=True)
    for repo, n, name in [
        ("ash12321/seedream-4.5-generated-2k", 300, "seedream45"),
        ("exdysa/nano-banana-pro-generated-1k-clone", 300, "nano_banana"),
        ("LukasT9/Flux-1-Dev-Images-1k", 300, "flux_dev"),
        ("LukasT9/Flux-1-Schnell-Images-1k", 300, "flux_schnell"),
    ]:
        extract_and_cache(f"fake_{name}",
                          stream_hf_images(repo, max_n=n),
                          art_ext, learned_ext)

    # === FAKE: CivitAI (all generators) ===
    print("\n--- FAKE (CivitAI) ---", flush=True)
    for gen_dir in ["flux", "sdxl", "pony", "illustrious", "sd3", "sd35", "recraft", "gemini"]:
        p = CIVITAI / gen_dir
        if p.exists() and any(p.iterdir()):
            extract_and_cache(f"fake_civitai_{gen_dir}",
                              load_paths_as_images(p, recursive=False, max_n=300),
                              art_ext, learned_ext)

    # === FAKE: Hemg AI ===
    print("\n--- FAKE (Hemg AI) ---", flush=True)
    extract_and_cache("fake_hemg_ai",
                      stream_hf_images("Hemg/AI-Generated-vs-Real-Images-Datasets", max_n=800, label_filter=0),
                      art_ext, learned_ext)

    print("\n  ALL EXTRACTION COMPLETE", flush=True)
    print(f"  Cache dir: {CACHE_DIR}", flush=True)
    print(f"  Files: {len(list(CACHE_DIR.glob('*.npz')))}", flush=True)


def train_and_test():
    """Load cached features, train ensemble, test on tellif."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    print("=" * 60, flush=True)
    print("  TRAIN + TEST (from cached features)", flush=True)
    print("=" * 60, flush=True)

    # Load all cached features
    real_parts, fake_parts = [], []
    for f in sorted(CACHE_DIR.glob("real_*.npz")):
        data = np.load(f)
        real_parts.append(data["X"])
        print(f"  Real: {f.stem} ({len(data['X'])})", flush=True)
    for f in sorted(CACHE_DIR.glob("fake_*.npz")):
        data = np.load(f)
        fake_parts.append(data["X"])
        print(f"  Fake: {f.stem} ({len(data['X'])})", flush=True)

    X_real = np.vstack(real_parts)
    X_fake = np.vstack(fake_parts)
    n = min(len(X_real), len(X_fake))
    X_train = np.vstack([X_real[:n], X_fake[:n]])
    y_train = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"\n  Training: {len(X_train)} ({n}/class)", flush=True)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for fold, (tr, te) in enumerate(skf.split(X_train, y_train)):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
                                num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED)
        m.fit(X_train[tr], y_train[tr])
        p = m.predict_proba(X_train[te])[:, 1]
        acc = accuracy_score(y_train[te], (p > 0.5).astype(int))
        accs.append(acc)
        print(f"  Fold {fold+1}: {acc:.4f}", flush=True)
    print(f"  Mean: {np.mean(accs):.4f}", flush=True)

    # Train ensemble
    W_LGBM, W_RF, W_SVM = 0.4, 0.1, 0.5
    HIGH_T, LOW_T = 0.80, 0.20

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

    # Test tellif
    print("\n" + "=" * 60, flush=True)
    print("  TELLIF RESULTS", flush=True)
    print("=" * 60, flush=True)

    from datasets import load_dataset
    ds_tellif = load_dataset("tellif/ai_vs_real_image_semantically_similar", split="test")
    label_names = ds_tellif.features["label"].names

    art_ext, learned_ext = get_extractors()
    results = {}

    for i, gen_name in enumerate(label_names):
        indices = [j for j in range(len(ds_tellif)) if ds_tellif[j]["label"] == i]
        if not indices:
            continue
        images = [ds_tellif[j]["image"].convert("RGB") for j in indices]

        rows = [extract_one(art_ext, learned_ext, img) for img in images]
        X_gen = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        pred, prob = predict(X_gen)

        if gen_name == "real":
            fp = (pred == 1).sum()
            print(f"  {gen_name:45s}  n={len(X_gen):3d}  GENUINE={(pred==0).sum()}  UNC={(pred==-1).sum()}  FP={fp}  FP={fp/len(X_gen):.1%}", flush=True)
            results[gen_name] = {"n": len(X_gen), "fp": int(fp), "fp_rate": float(fp / len(X_gen))}
        else:
            det = (pred == 1).sum()
            rate = det / len(X_gen)
            print(f"  {gen_name:45s}  n={len(X_gen):3d}  DET={det}  UNC={(pred==-1).sum()}  MISS={(pred==0).sum()}  rate={rate:.1%}", flush=True)
            results[gen_name] = {"n": len(X_gen), "detected": int(det), "rate": float(rate)}

    out = Path(__file__).parent.parent / "results" / "retrain_weak_gens_results.json"
    with open(out, "w") as f:
        json.dump({"cv_acc": float(np.mean(accs)), "n_train": int(len(X_train)), "tellif": results}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_and_test()
    else:
        extract_all()
