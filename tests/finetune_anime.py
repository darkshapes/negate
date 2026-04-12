"""Fine-tune ConvNeXt-Tiny last stage for anime real vs AI detection.

Freeze stages 0-2, train stage 3 + classification head on anime data.
CPU training — slow but feasible for our data size (~2600 images).
"""

from __future__ import annotations

import os
import warnings

os.environ["HF_HOME"] = "D:/Projects/negate/negate/.cache/huggingface"
warnings.filterwarnings("ignore")

import numpy as np
import timm
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE = Path("D:/Projects/negate/negate/.datasets/imaginet/extracted")
CIVITAI = Path("D:/Projects/negate/negate/.datasets/civitai")


class AnimeDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)


def load_paths(path, recursive, max_n):
    rng = np.random.RandomState(SEED)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = list(path.rglob("*") if recursive else path.iterdir())
    files = [f for f in files if f.suffix.lower() in exts]
    if len(files) > max_n:
        files = list(rng.choice(files, max_n, replace=False))
    imgs = []
    for f in files:
        try:
            imgs.append(Image.open(f).convert("RGB"))
        except Exception:
            pass
    return imgs


def main():
    print("=" * 60, flush=True)
    print("  FINE-TUNE ConvNeXt-Tiny for Anime Detection", flush=True)
    print("  Freeze stages 0-2, train stage 3 + head", flush=True)
    print("=" * 60, flush=True)

    # Load model
    model = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=2)
    transform = timm.data.create_transform(
        **timm.data.resolve_data_config(model.pretrained_cfg)
    )

    # Freeze everything except stage 3 and head
    for name, param in model.named_parameters():
        if "stages.3" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)", flush=True)

    # Load data
    print("\n--- Loading data ---", flush=True)
    real_imgs = []

    # Real anime from animesfw (1000)
    ds = load_dataset("latentcat/animesfw", split="train", streaming=True)
    count = 0
    for s in tqdm(ds, desc="Real anime", total=1000):
        if count >= 1000:
            break
        tags = s.get("tags", "")
        if "ai" in tags.lower():
            continue
        try:
            real_imgs.append(s["image"].convert("RGB"))
            count += 1
        except Exception:
            pass
    print(f"  Real anime: {len(real_imgs)}", flush=True)

    # Real WikiArt (500)
    wiki_imgs = load_paths(BASE / "wikiart", recursive=True, max_n=500)
    real_imgs.extend(wiki_imgs)
    print(f"  + WikiArt: {len(wiki_imgs)}, total real: {len(real_imgs)}", flush=True)

    # Fake anime (CivitAI pony + illustrious + AnimagineXL)
    fake_imgs = []
    for gen in ["pony", "illustrious"]:
        imgs = load_paths(CIVITAI / gen, recursive=False, max_n=400)
        fake_imgs.extend(imgs)
    animagine = load_paths(BASE / "animaginexl_paintings_fake", recursive=False, max_n=100)
    fake_imgs.extend(animagine)

    # Add Flux and nano-banana for diversity
    for repo, n in [("LukasT9/Flux-1-Dev-Images-1k", 200), ("bitmind/nano-banana", 200)]:
        ds2 = load_dataset(repo, split="train", streaming=True)
        c = 0
        for s in ds2:
            if c >= n:
                break
            try:
                fake_imgs.append(s["image"].convert("RGB"))
                c += 1
            except Exception:
                pass
    print(f"  Fake: {len(fake_imgs)}", flush=True)

    # Balance
    n = min(len(real_imgs), len(fake_imgs))
    all_imgs = real_imgs[:n] + fake_imgs[:n]
    all_labels = [0] * n + [1] * n
    print(f"  Total: {len(all_imgs)} ({n}/class)", flush=True)

    # Split train/val
    indices = np.random.permutation(len(all_imgs))
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = AnimeDataset(
        [all_imgs[i] for i in train_idx],
        [all_labels[i] for i in train_idx],
        transform,
    )
    val_ds = AnimeDataset(
        [all_imgs[i] for i in val_idx],
        [all_labels[i] for i in val_idx],
        transform,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Train
    print(f"\n--- Training ({len(train_ds)} train, {len(val_ds)} val) ---", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(5):
        losses = []
        correct = 0
        total = 0
        for batch_imgs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels.long()).sum().item()
            total += len(batch_labels)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_fp = 0
        val_real = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_labels.long()).sum().item()
                val_total += len(batch_labels)
                # FP: real labeled as AI
                real_mask = batch_labels == 0
                val_fp += ((preds == 1) & real_mask).sum().item()
                val_real += real_mask.sum().item()
        model.train()

        val_acc = val_correct / val_total
        fp_rate = val_fp / val_real if val_real > 0 else 0
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f} train_acc={correct/total:.4f} val_acc={val_acc:.4f} val_FP={fp_rate:.3f}", flush=True)

    # Test on held-out anime
    print("\n--- Testing on held-out anime ---", flush=True)
    model.eval()

    ds3 = load_dataset("latentcat/animesfw", split="train", streaming=True)
    test_fp = 0
    test_total = 0
    skip = 0
    with torch.no_grad():
        for s in tqdm(ds3, desc="Test anime", total=1200):
            tags = s.get("tags", "")
            if "ai" in tags.lower():
                continue
            skip += 1
            if skip <= 1000:
                continue
            if test_total >= 200:
                break
            try:
                img = s["image"].convert("RGB")
                tensor = transform(img).unsqueeze(0)
                output = model(tensor)
                pred = output.argmax(dim=1).item()
                if pred == 1:  # flagged as AI
                    test_fp += 1
                test_total += 1
            except Exception:
                pass

    print(f"\nAnime FP (fine-tuned ConvNeXt): {test_fp}/{test_total} ({test_fp/test_total*100:.1f}%)", flush=True)

    # Save model
    out_path = Path("D:/Projects/negate/negate/models/convnext_anime_finetuned.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
