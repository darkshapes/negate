"""Multi-dataset benchmark for artwork feature extraction + LightGBM classification."""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "D:/Projects/negate/negate/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/Projects/negate/negate/.cache/huggingface/datasets"

import json
import time
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_features_batch(images: list, extractor) -> NDArray:
    """Extract features from a list of PIL images, return (N, D) array."""
    rows = []
    for img in tqdm(images, desc="Extracting features", leave=False):
        try:
            feats = extractor(img)
            rows.append(list(feats.values()))
        except Exception:
            rows.append([0.0] * 148)
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def train_and_evaluate(X_train: NDArray, y_train: NDArray, X_test: NDArray, y_test: NDArray) -> dict:
    """Train LightGBM and return accuracy + AUC."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        n_jobs=1,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": round(acc, 4), "auc": round(auc, 4)}


def cross_validate(X: NDArray, y: NDArray, n_splits: int = 5) -> dict:
    """5-fold stratified CV, return pooled accuracy + AUC."""
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, n_jobs=1, verbose=-1, random_state=42,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)
    acc = accuracy_score(all_y_true, all_y_pred)
    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": round(acc, 4), "auc": round(auc, 4)}


def sample_balanced(images: list, labels: list, max_per_class: int) -> tuple[list, list]:
    """Subsample up to max_per_class per unique label value."""
    from collections import defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        buckets[l].append(i)
    rng = np.random.RandomState(42)
    selected: list[int] = []
    for cls, idxs in sorted(buckets.items()):
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, max_per_class, replace=False).tolist()
        selected.extend(idxs)
    rng.shuffle(selected)
    return [images[i] for i in selected], [labels[i] for i in selected]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_defactify(extractor, max_per_gen: int = 200) -> tuple[NDArray, NDArray, dict]:
    """Load Defactify, sample up to max_per_gen real+fake per generator.

    Uses Rajarshi-Roy-research/Defactify_Image_Dataset with:
      - Label_B: 0=real, 1=SD2.1, 2=SDXL, 3=SD3, 4=Midjourney v6, 5=DALL-E 3
      - Image: PIL image column
    """
    from collections import defaultdict
    from datasets import load_dataset, Image as HFImage

    GENERATOR_NAMES = {1: "SD_2.1", 2: "SDXL", 3: "SD_3", 4: "Midjourney_v6", 5: "DALL-E_3"}

    print("\n=== Loading Defactify ===")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", split="train")
    ds = ds.cast_column("Image", HFImage(decode=True, mode="RGB"))
    print(f"  Total samples: {len(ds)}")

    # Group by Label_B
    label_b = ds["Label_B"]
    gen_buckets: dict[str, list[int]] = defaultdict(list)
    real_indices: list[int] = []
    for i, lb in enumerate(label_b):
        if lb == 0:
            real_indices.append(i)
        elif lb in GENERATOR_NAMES:
            gen_buckets[GENERATOR_NAMES[lb]].append(i)

    generators = sorted(gen_buckets.keys())
    print(f"  Real images: {len(real_indices)}")
    print(f"  Generators: {generators}")
    for g in generators:
        print(f"    {g}: {len(gen_buckets[g])}")

    # Sample real + per-generator fake
    rng = np.random.RandomState(42)
    selected_indices: list[int] = []
    selected_labels: list[int] = []
    gen_indices: dict[str, list[int]] = {}

    # Sample real images
    real_sample = rng.choice(real_indices, size=min(max_per_gen, len(real_indices)), replace=False).tolist()
    start = len(selected_indices)
    selected_indices.extend(real_sample)
    selected_labels.extend([0] * len(real_sample))
    gen_indices["real"] = list(range(start, len(selected_indices)))

    # Sample fake per generator
    for g in generators:
        idxs = gen_buckets[g]
        if len(idxs) > max_per_gen:
            idxs = rng.choice(idxs, max_per_gen, replace=False).tolist()
        start = len(selected_indices)
        selected_indices.extend(idxs)
        selected_labels.extend([1] * len(idxs))
        gen_indices[g] = list(range(start, len(selected_indices)))

    print(f"  Selected {len(selected_indices)} samples (real={sum(1 for l in selected_labels if l==0)}, fake={sum(1 for l in selected_labels if l==1)})")
    import gc

    # Extract features one at a time to avoid holding all images in memory
    rows = []
    for idx in tqdm(selected_indices, desc="Defactify: extract features"):
        try:
            img = ds[int(idx)]["Image"]
            feats = extractor(img)
            rows.append(list(feats.values()))
            del img
        except Exception:
            rows.append([0.0] * 148)

    del ds
    gc.collect()

    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(selected_labels, dtype=np.int32)

    meta = {"generators": generators, "gen_indices": gen_indices}
    return X, y, meta


def load_hemg(extractor, max_per_class: int = 1000) -> tuple[NDArray, NDArray]:
    """Load Hemg AI-art dataset. label: 0=AI, 1=real."""
    from collections import defaultdict
    from datasets import load_dataset
    import gc

    print("\n=== Loading Hemg (AI art vs real art) ===")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    print(f"  Total samples: {len(ds)}")

    # Read only labels column (fast, avoids decoding images)
    all_labels = ds["label"]  # fetches entire label column as list
    label_buckets: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(all_labels):
        label_buckets[lbl].append(i)

    for k, v in sorted(label_buckets.items()):
        print(f"  Label {k}: {len(v)} samples")

    # Sample indices first
    rng = np.random.RandomState(42)
    selected_indices: list[int] = []
    selected_labels: list[int] = []
    for cls in sorted(label_buckets.keys()):
        idxs = label_buckets[cls]
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, max_per_class, replace=False).tolist()
        selected_indices.extend(idxs)
        selected_labels.extend([cls] * len(idxs))

    print(f"  Selected {len(selected_indices)} samples")

    # Extract features one at a time
    rows = []
    final_labels = []
    for idx, lbl in tqdm(zip(selected_indices, selected_labels), total=len(selected_indices), desc="Hemg: extract features"):
        try:
            img = ds[idx]["image"]
            feats = extractor(img)
            rows.append(list(feats.values()))
            final_labels.append(lbl)
            del img
        except Exception:
            rows.append([0.0] * 148)
            final_labels.append(lbl)

    del ds
    gc.collect()

    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(final_labels, dtype=np.int32)
    return X, y


def load_tellif(extractor, max_per_class: int = 200) -> tuple[NDArray, NDArray, list[str]]:
    """Load tellif semantically similar dataset."""
    from collections import defaultdict
    from datasets import load_dataset
    import gc

    print("\n=== Loading tellif (semantically similar) ===")
    ds = load_dataset("tellif/ai_vs_real_image_semantically_similar", split="test")
    print(f"  Total samples: {len(ds)}")

    label_names = ds.features["label"].names
    print(f"  Label names: {label_names}")

    # First pass: read labels only
    all_labels_raw = [ds[i]["label"] for i in range(len(ds))]

    unique, counts = np.unique(all_labels_raw, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]} ({u}): {c}")

    # Map to binary: find which labels are "real" vs AI-generated
    real_keywords = ["real", "photograph", "photo", "human", "genuine", "original"]
    real_label_ids = set()
    for lid, name in enumerate(label_names):
        if any(kw in name.lower() for kw in real_keywords):
            real_label_ids.add(lid)
    print(f"  Detected real labels: {[label_names[i] for i in sorted(real_label_ids)]}")
    if not real_label_ids:
        # If no clear "real" label, treat the first label as real
        print("  WARNING: No clear 'real' label found. Treating label 0 as real.")
        real_label_ids = {0}

    binary_labels = [0 if l in real_label_ids else 1 for l in all_labels_raw]

    # Sample indices per binary class
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, bl in enumerate(binary_labels):
        buckets[bl].append(i)
    rng = np.random.RandomState(42)
    selected_indices: list[int] = []
    selected_binary: list[int] = []
    for cls in sorted(buckets.keys()):
        idxs = buckets[cls]
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, max_per_class, replace=False).tolist()
        selected_indices.extend(idxs)
        selected_binary.extend([cls] * len(idxs))
    print(f"  Selected {len(selected_indices)} samples (binary: 0=real, 1=AI)")

    # Extract features one at a time
    rows = []
    final_labels = []
    for idx, lbl in tqdm(zip(selected_indices, selected_binary), total=len(selected_indices), desc="tellif: extract features"):
        try:
            img = ds[idx]["image"]
            if img is not None:
                feats = extractor(img)
                rows.append(list(feats.values()))
                final_labels.append(lbl)
                del img
        except Exception:
            continue

    del ds
    gc.collect()
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(final_labels, dtype=np.int32)
    return X, y, label_names


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_1_baselines(X_def, y_def, X_hemg, y_hemg, X_tellif, y_tellif) -> dict:
    """Exp 1: Single-dataset baselines with 5-fold CV."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Single-dataset baselines (5-fold CV)")
    print("=" * 70)

    results = {}
    for name, X, y in [("Defactify", X_def, y_def), ("Hemg", X_hemg, y_hemg), ("tellif", X_tellif, y_tellif)]:
        print(f"\n  {name}: {X.shape[0]} samples, {X.shape[1]} features")
        r = cross_validate(X, y)
        results[name] = r
        print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  +--------------+----------+--------+")
    print("  | Dataset      | Accuracy |   AUC  |")
    print("  +--------------+----------+--------+")
    for name, r in results.items():
        print(f"  | {name:<12} | {r['accuracy']:.4f}   | {r['auc']:.4f} |")
    print("  +--------------+----------+--------+")
    return results


def experiment_2_cross_dataset(X_def, y_def, X_hemg, y_hemg, X_tellif, y_tellif) -> dict:
    """Exp 2: Cross-dataset generalization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-dataset generalization")
    print("=" * 70)

    # Normalize Hemg labels: 0=AI->1(fake), 1=real->0(real) to match Defactify convention
    # Defactify: 0=real, 1=fake
    # Hemg: 0=AI, 1=real  -> need to flip
    y_hemg_norm = 1 - y_hemg
    # tellif: already 0=real, 1=AI

    results = {}

    # Train Defactify -> test Hemg
    print("\n  Train Defactify -> Test Hemg")
    r = train_and_evaluate(X_def, y_def, X_hemg, y_hemg_norm)
    results["Defactify->Hemg"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # Train Hemg -> test Defactify
    print("\n  Train Hemg -> Test Defactify")
    r = train_and_evaluate(X_hemg, y_hemg_norm, X_def, y_def)
    results["Hemg->Defactify"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # Combined train -> test each
    X_combined = np.vstack([X_def, X_hemg])
    y_combined = np.concatenate([y_def, y_hemg_norm])

    print("\n  Train Combined -> Test Defactify")
    r = train_and_evaluate(X_combined, y_combined, X_def, y_def)
    results["Combined->Defactify"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  Train Combined -> Test Hemg")
    r = train_and_evaluate(X_combined, y_combined, X_hemg, y_hemg_norm)
    results["Combined->Hemg"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  Train Combined -> Test tellif")
    r = train_and_evaluate(X_combined, y_combined, X_tellif, y_tellif)
    results["Combined->tellif"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  +------------------------+----------+--------+")
    print("  | Transfer               | Accuracy |   AUC  |")
    print("  +------------------------+----------+--------+")
    for name, r in results.items():
        print(f"  | {name:<22} | {r['accuracy']:.4f}   | {r['auc']:.4f} |")
    print("  +------------------------+----------+--------+")
    return results


def experiment_3_generator_diversity(X_def, y_def, meta_def) -> dict:
    """Exp 3: Generator diversity impact using Defactify."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Generator diversity impact (Defactify)")
    print("=" * 70)

    generators = meta_def["generators"]
    gen_indices = meta_def["gen_indices"]
    real_idx = np.array(gen_indices["real"])  # shared real block
    results = {}

    # Build per-generator fake-only index arrays
    gen_fake_arrays = {}
    for g in generators:
        gen_fake_arrays[g] = np.array(gen_indices[g])

    rng = np.random.RandomState(42)
    gen_order = list(generators)
    rng.shuffle(gen_order)
    print(f"  Generator order: {gen_order}")

    # Split real indices proportionally for train/test
    n_gens = len(generators)

    # Progressive: train on N generators, test on rest
    for n_train in range(1, n_gens):
        train_gens = gen_order[:n_train]
        test_gens = gen_order[n_train:]

        # Split real images proportionally
        real_shuffled = rng.permutation(real_idx)
        n_real_train = int(len(real_shuffled) * n_train / n_gens)
        real_train = real_shuffled[:n_real_train]
        real_test = real_shuffled[n_real_train:]

        train_fake = np.concatenate([gen_fake_arrays[g] for g in train_gens])
        test_fake = np.concatenate([gen_fake_arrays[g] for g in test_gens])

        train_idx = np.concatenate([real_train, train_fake])
        test_idx = np.concatenate([real_test, test_fake])

        X_tr, y_tr = X_def[train_idx], y_def[train_idx]
        X_te, y_te = X_def[test_idx], y_def[test_idx]

        label = f"{n_train}_gen_train"
        print(f"\n  Train on {n_train} gen ({', '.join(train_gens)}) -> Test on ({', '.join(test_gens)})")
        print(f"    Train: {len(X_tr)} (real={np.sum(y_tr==0)}, fake={np.sum(y_tr==1)})")
        print(f"    Test:  {len(X_te)} (real={np.sum(y_te==0)}, fake={np.sum(y_te==1)})")
        r = train_and_evaluate(X_tr, y_tr, X_te, y_te)
        r["train_generators"] = train_gens
        r["test_generators"] = test_gens
        results[label] = r
        print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # All 5 generators: 5-fold CV
    print(f"\n  All {n_gens} generators: 5-fold CV")
    r = cross_validate(X_def, y_def)
    r["train_generators"] = generators
    r["test_generators"] = generators
    results[f"{n_gens}_gen_cv"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  +----------------+----------+--------+")
    print("  | # Generators   | Accuracy |   AUC  |")
    print("  +----------------+----------+--------+")
    for name, r in results.items():
        print(f"  | {name:<14} | {r['accuracy']:.4f}   | {r['auc']:.4f} |")
    print("  +----------------+----------+--------+")
    return results


def experiment_4_domain_transfer(X_def, y_def, X_hemg, y_hemg) -> dict:
    """Exp 4: Art vs photos domain transfer."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Art vs Photos domain transfer")
    print("=" * 70)

    # Normalize Hemg: 0=AI->1, 1=real->0
    y_hemg_norm = 1 - y_hemg

    results = {}

    # Photos -> Art
    print("\n  Train Defactify (photos) -> Test Hemg (art)")
    r = train_and_evaluate(X_def, y_def, X_hemg, y_hemg_norm)
    results["Photos->Art"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # Art -> Photos
    print("\n  Train Hemg (art) -> Test Defactify (photos)")
    r = train_and_evaluate(X_hemg, y_hemg_norm, X_def, y_def)
    results["Art->Photos"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # Same-domain baselines for comparison
    print("\n  Baseline: Defactify 5-fold CV (same domain)")
    r = cross_validate(X_def, y_def)
    results["Photos_self_CV"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    print("\n  Baseline: Hemg 5-fold CV (same domain)")
    r = cross_validate(X_hemg, y_hemg_norm)
    results["Art_self_CV"] = r
    print(f"    Accuracy: {r['accuracy']:.4f}  AUC: {r['auc']:.4f}")

    # Compute domain gap
    photo_self = results["Photos_self_CV"]["accuracy"]
    art_self = results["Art_self_CV"]["accuracy"]
    p2a = results["Photos->Art"]["accuracy"]
    a2p = results["Art->Photos"]["accuracy"]
    print(f"\n  Domain gap (Photos->Art): {photo_self:.4f} -> {p2a:.4f} (delta: {photo_self - p2a:+.4f})")
    print(f"  Domain gap (Art->Photos): {art_self:.4f} -> {a2p:.4f} (delta: {art_self - a2p:+.4f})")

    print("\n  +------------------+----------+--------+")
    print("  | Transfer         | Accuracy |   AUC  |")
    print("  +------------------+----------+--------+")
    for name, r in results.items():
        print(f"  | {name:<16} | {r['accuracy']:.4f}   | {r['auc']:.4f} |")
    print("  +------------------+----------+--------+")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    from negate.extract.feature_artwork import ArtworkExtract
    extractor = ArtworkExtract()

    # Determine feature count
    from PIL import Image
    dummy = Image.new("RGB", (255, 255), color="gray")
    n_features = len(extractor(dummy))
    print(f"Feature count: {n_features}")

    # Load datasets
    X_def, y_def, meta_def = load_defactify(extractor, max_per_gen=200)
    X_hemg, y_hemg = load_hemg(extractor, max_per_class=1000)
    X_tellif, y_tellif, tellif_labels = load_tellif(extractor, max_per_class=200)

    print(f"\nDataset shapes:")
    print(f"  Defactify: X={X_def.shape}, y={y_def.shape} (real={np.sum(y_def==0)}, fake={np.sum(y_def==1)})")
    print(f"  Hemg:      X={X_hemg.shape}, y={y_hemg.shape} (AI={np.sum(y_hemg==0)}, real={np.sum(y_hemg==1)})")
    print(f"  tellif:    X={X_tellif.shape}, y={y_tellif.shape} (real={np.sum(y_tellif==0)}, AI={np.sum(y_tellif==1)})")

    # Run experiments
    results_all: dict = {"feature_count": n_features}

    results_all["exp1_baselines"] = experiment_1_baselines(X_def, y_def, X_hemg, y_hemg, X_tellif, y_tellif)
    results_all["exp2_cross_dataset"] = experiment_2_cross_dataset(X_def, y_def, X_hemg, y_hemg, X_tellif, y_tellif)
    results_all["exp3_generator_diversity"] = experiment_3_generator_diversity(X_def, y_def, meta_def)
    results_all["exp4_domain_transfer"] = experiment_4_domain_transfer(X_def, y_def, X_hemg, y_hemg)

    elapsed = time.time() - start_time
    results_all["elapsed_seconds"] = round(elapsed, 1)

    # Save results
    out_path = Path("D:/Projects/negate/negate/results/multidataset_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(_make_serializable(results_all), f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
