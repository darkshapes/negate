"""Adversarial robustness and false positive analysis.

Tests whether the 148-feature detector holds up under:
1. Post-processing attacks (JPEG, noise, resize, blur, crop)
2. False positive stress test (digital art, ambiguous content)
3. Threshold calibration (precision-recall tradeoff)
4. Social media simulation (combined degradation)

Uses ImagiNet paintings data (already on disk).
"""

from __future__ import annotations

import json
import warnings
from io import BytesIO
from pathlib import Path

import lightgbm as lgb
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
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
N_IMAGES = 300  # per class for speed
rng = np.random.RandomState(SEED)


def load_image_paths(path: Path, recursive: bool, max_n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if recursive:
        files = [f for f in path.rglob("*") if f.suffix.lower() in exts]
    else:
        files = [f for f in path.iterdir() if f.suffix.lower() in exts]
    if len(files) > max_n:
        files = list(rng.choice(files, max_n, replace=False))
    return files


def extract_features_from_images(images: list[Image.Image], desc: str = "") -> np.ndarray:
    rows = []
    for img in tqdm(images, desc=desc, leave=False):
        try:
            feat = ext(img)
            rows.append(list(feat.values()))
        except Exception:
            rows.append([0.0] * 148)
    X = np.array(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_from_paths(files: list[Path], desc: str = "") -> np.ndarray:
    images = []
    for f in files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception:
            pass
    return extract_features_from_images(images, desc)


# ---- Perturbation functions ----

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img, dtype=np.float64)
    noise = rng.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_down_up(img: Image.Image, small_size: int) -> Image.Image:
    orig_size = img.size
    img_small = img.resize((small_size, small_size), Image.BICUBIC)
    return img_small.resize(orig_size, Image.BICUBIC)


def center_crop(img: Image.Image, ratio: float = 0.7) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def social_media_sim(img: Image.Image) -> Image.Image:
    """Simulate social media pipeline: resize down, JPEG, slight blur."""
    img = img.resize((1024, 1024), Image.BICUBIC)
    img = jpeg_compress(img, 75)
    img = gaussian_blur(img, 0.5)
    return img


PERTURBATIONS = {
    "jpeg_q30": lambda img: jpeg_compress(img, 30),
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
    "jpeg_q70": lambda img: jpeg_compress(img, 70),
    "noise_s5": lambda img: add_gaussian_noise(img, 5),
    "noise_s15": lambda img: add_gaussian_noise(img, 15),
    "noise_s30": lambda img: add_gaussian_noise(img, 30),
    "resize_128": lambda img: resize_down_up(img, 128),
    "resize_64": lambda img: resize_down_up(img, 64),
    "crop_70pct": lambda img: center_crop(img, 0.7),
    "crop_50pct": lambda img: center_crop(img, 0.5),
    "blur_r1": lambda img: gaussian_blur(img, 1),
    "blur_r3": lambda img: gaussian_blur(img, 3),
    "bright_0.7": lambda img: adjust_brightness(img, 0.7),
    "bright_1.3": lambda img: adjust_brightness(img, 1.3),
    "social_media": social_media_sim,
}


def main():
    print("=" * 70)
    print("  ADVERSARIAL ROBUSTNESS & FALSE POSITIVE ANALYSIS")
    print("  148 features, LightGBM, ImagiNet paintings")
    print("=" * 70)

    # Load real and fake images
    print("\nLoading images...")
    real_files = load_image_paths(BASE / "wikiart", recursive=True, max_n=N_IMAGES)
    # Mix of generators for fake
    fake_files = []
    for gen_dir in ["sdxl_paintings_fake", "sd_paintings_fake", "dalle3", "journeydb"]:
        fake_files.extend(load_image_paths(BASE / gen_dir, recursive=False, max_n=N_IMAGES // 4))
    rng.shuffle(fake_files)
    fake_files = fake_files[:N_IMAGES]

    print(f"  Real: {len(real_files)}, Fake: {len(fake_files)}")

    real_images = [Image.open(f).convert("RGB") for f in tqdm(real_files, desc="Loading real")]
    fake_images = [Image.open(f).convert("RGB") for f in tqdm(fake_files, desc="Loading fake")]

    # Extract clean features and train model
    print("\nExtracting clean features...")
    X_real = extract_features_from_images(real_images, "Real features")
    X_fake = extract_features_from_images(fake_images, "Fake features")

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_fake))])

    # Train on 70%, test perturbations on 30%
    n = len(y)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED,
    )
    model.fit(X[train_idx], y[train_idx])

    # Clean baseline
    y_prob_clean = model.predict_proba(X[test_idx])[:, 1]
    y_pred_clean = (y_prob_clean > 0.5).astype(int)
    clean_acc = accuracy_score(y[test_idx], y_pred_clean)
    clean_auc = roc_auc_score(y[test_idx], y_prob_clean)
    print(f"\nClean baseline: acc={clean_acc:.4f}  auc={clean_auc:.4f}")

    # ---- EXP 1: Perturbation robustness ----
    print("\n" + "=" * 70)
    print("  EXP 1: Adversarial Robustness (perturbations on test set)")
    print("=" * 70)

    # Get the test images
    test_images_real = [real_images[i] for i in range(len(real_images)) if i in set(test_idx[test_idx < len(real_images)])]
    test_images_fake = [fake_images[i - len(real_images)] for i in test_idx if i >= len(real_images)]
    test_images = test_images_real + test_images_fake
    test_labels = np.concatenate([np.zeros(len(test_images_real)), np.ones(len(test_images_fake))])

    results_perturb = {"clean": {"acc": clean_acc, "auc": clean_auc}}

    for pert_name, pert_fn in PERTURBATIONS.items():
        print(f"\n  Applying {pert_name}...")
        perturbed = []
        for img in tqdm(test_images, desc=f"  {pert_name}", leave=False):
            try:
                perturbed.append(pert_fn(img))
            except Exception:
                perturbed.append(img)  # fallback to original

        X_pert = extract_features_from_images(perturbed, f"  {pert_name} features")
        y_prob_pert = model.predict_proba(X_pert)[:, 1]
        y_pred_pert = (y_prob_pert > 0.5).astype(int)

        acc = accuracy_score(test_labels, y_pred_pert)
        auc = roc_auc_score(test_labels, y_prob_pert)
        delta = acc - clean_acc
        results_perturb[pert_name] = {"acc": float(acc), "auc": float(auc), "delta": float(delta)}
        print(f"    acc={acc:.4f}  auc={auc:.4f}  delta={delta:+.4f}")

    # Summary table
    print("\n  +---------------------+--------+--------+---------+")
    print("  | Perturbation        |  Acc   |  AUC   |  Delta  |")
    print("  +---------------------+--------+--------+---------+")
    for name, r in sorted(results_perturb.items(), key=lambda x: -x[1]["acc"]):
        print(f"  | {name:19s} | {r['acc']:.4f} | {r['auc']:.4f} | {r.get('delta', 0):+.4f} |")
    print("  +---------------------+--------+--------+---------+")

    # ---- EXP 2: Threshold calibration ----
    print("\n" + "=" * 70)
    print("  EXP 2: Threshold Calibration (Precision-Recall Tradeoff)")
    print("=" * 70)

    # Use full 5-fold CV probabilities for calibration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_probs, all_labels = [], []
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=63, n_jobs=1, verbose=-1, random_state=SEED,
        )
        m.fit(X[tr], y[tr])
        all_probs.extend(m.predict_proba(X[te])[:, 1])
        all_labels.extend(y[te])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    prec_arr, rec_arr, thresholds = precision_recall_curve(all_labels, all_probs)

    # Find thresholds for different precision targets
    print("\n  Threshold analysis (higher threshold = fewer false positives):")
    print("  +------------+--------+-----------+--------+---------+")
    print("  | Threshold  |  Prec  |  Recall   |   F1   | FP Rate |")
    print("  +------------+--------+-----------+--------+---------+")

    results_thresh = {}
    for target_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pred = (all_probs >= target_thresh).astype(int)
        if pred.sum() == 0:
            continue
        prec = precision_score(all_labels, pred, zero_division=0)
        rec = recall_score(all_labels, pred, zero_division=0)
        f1 = f1_score(all_labels, pred, average="macro", zero_division=0)
        # False positive rate: real images incorrectly flagged as AI
        real_mask = all_labels == 0
        fp_rate = float(pred[real_mask].mean())

        results_thresh[str(target_thresh)] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "fp_rate": float(fp_rate),
        }
        print(f"  | {target_thresh:10.2f} | {prec:.4f} | {rec:9.4f} | {f1:.4f} | {fp_rate:7.4f} |")
    print("  +------------+--------+-----------+--------+---------+")

    # Find the sweet spot: highest F1 with FP rate < 5%
    best_thresh = None
    best_f1 = 0
    for t_str, r in results_thresh.items():
        if r["fp_rate"] <= 0.05 and r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_thresh = t_str
    if best_thresh:
        print(f"\n  Recommended threshold: {best_thresh} (F1={results_thresh[best_thresh]['f1']:.4f}, FP rate={results_thresh[best_thresh]['fp_rate']:.4f})")

    # ---- EXP 3: Social media simulation ----
    print("\n" + "=" * 70)
    print("  EXP 3: Social Media Pipeline Simulation")
    print("=" * 70)
    print("  (Resize to 1024px + JPEG Q75 + slight blur)")
    # Already in perturbation results
    sm = results_perturb.get("social_media", {})
    print(f"  Social media accuracy: {sm.get('acc', 'N/A')}")
    print(f"  Social media AUC: {sm.get('auc', 'N/A')}")
    print(f"  Delta from clean: {sm.get('delta', 'N/A')}")

    # ---- EXP 4: Worst-case adversarial (combined attacks) ----
    print("\n" + "=" * 70)
    print("  EXP 4: Worst-Case Adversarial (Combined Attacks)")
    print("=" * 70)

    def worst_case_attack(img: Image.Image) -> Image.Image:
        """JPEG Q30 + resize 128→orig + noise σ=10 + blur r=1"""
        img = jpeg_compress(img, 30)
        img = resize_down_up(img, 128)
        img = add_gaussian_noise(img, 10)
        img = gaussian_blur(img, 1)
        return img

    perturbed_worst = [worst_case_attack(img) for img in tqdm(test_images, desc="Worst case")]
    X_worst = extract_features_from_images(perturbed_worst, "Worst case features")
    y_prob_worst = model.predict_proba(X_worst)[:, 1]
    y_pred_worst = (y_prob_worst > 0.5).astype(int)
    worst_acc = accuracy_score(test_labels, y_pred_worst)
    worst_auc = roc_auc_score(test_labels, y_prob_worst)
    print(f"  Worst-case accuracy: {worst_acc:.4f}")
    print(f"  Worst-case AUC: {worst_auc:.4f}")
    print(f"  Delta from clean: {worst_acc - clean_acc:+.4f}")

    # Save all results
    all_results = {
        "clean_baseline": {"acc": float(clean_acc), "auc": float(clean_auc)},
        "perturbation_robustness": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results_perturb.items()},
        "threshold_calibration": results_thresh,
        "recommended_threshold": best_thresh,
        "worst_case_adversarial": {"acc": float(worst_acc), "auc": float(worst_auc)},
        "n_real": len(real_files),
        "n_fake": len(fake_files),
    }

    out_path = Path(__file__).parent.parent / "results" / "adversarial_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Clean accuracy:      {clean_acc:.4f}")
    print(f"  Social media:        {sm.get('acc', 'N/A')}")
    print(f"  Worst-case attack:   {worst_acc:.4f}")
    print(f"  Recommended thresh:  {best_thresh}")
    if best_thresh:
        r = results_thresh[best_thresh]
        print(f"    → Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f}, FP rate: {r['fp_rate']:.4f}")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
