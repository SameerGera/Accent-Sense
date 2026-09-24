"""
AccentSense — Local DirectML Training Pipeline
Optimized for Windows laptops with AMD Radeon (RDNA / 860M), Intel Arc, or NVIDIA GPUs via DirectML.

Usage:
    python train_local_directml.py --epochs 15 --batch_size 4
"""

import argparse
import json
import os
import sys
from collections import Counter

# Ensure Backend directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.vctk_dataset import UKAccentDataset, collate_pad, load_manifest, save_manifest
from src.models.wavlm_classifier import WavLMAccentClassifier

# Target 6 UK Regional Accent Classes
CLASSES = ("RP", "Scottish", "Welsh", "Northern", "West_Midlands", "Irish")
N = len(CLASSES)  # 6


def get_dml_device():
    """Detects and returns the best hardware device: DirectML (AMD/Intel GPU) > CUDA > CPU."""
    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            print("=" * 65)
            print("✓ Hardware Acceleration: DirectML (DirectX 12 GPU)")
            print(f"  Device: {dml_device}")
            print("=" * 65)
            return dml_device
    except ImportError:
        pass

    if torch.cuda.is_available():
        print("✓ Hardware Acceleration: NVIDIA CUDA")
        return torch.device("cuda")

    print("⚠ DirectML / CUDA not detected. Falling back to multi-threaded CPU.")
    return torch.device("cpu")


def sanitize_manifests(classes):
    """Guarantees class_idx is strictly within [0, len(classes)-1] and prunes unused labels."""
    for split_name in ["train", "val", "test"]:
        p = os.path.join(CURRENT_DIR, "data", "manifests", f"{split_name}.json")
        if os.path.exists(p):
            data = load_manifest(p)
            valid = []
            modified = False
            for item in data:
                if item["label"] not in classes:
                    modified = True
                    continue
                correct_idx = classes.index(item["label"])
                if item.get("class_idx") != correct_idx:
                    item["class_idx"] = correct_idx
                    modified = True
                valid.append(item)
            if modified:
                print(f"✓ Sanitized {split_name}.json ({len(valid)} samples mapped to 6 classes)")
                save_manifest(valid, p)


def main():
    parser = argparse.ArgumentParser(description="Train AccentSense with DirectML on local laptop.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 recommended for laptop iGPUs)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for head")
    parser.add_argument("--unfreeze_top_k", type=int, default=2, help="Number of top encoder layers to fine-tune")
    parser.add_argument("--output_dir", type=str, default=os.path.join(CURRENT_DIR, "checkpoints"))
    args = parser.parse_args()

    device = get_dml_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Validate manifests exist
    train_manifest_path = os.path.join(CURRENT_DIR, "data", "manifests", "train.json")
    val_manifest_path = os.path.join(CURRENT_DIR, "data", "manifests", "val.json")

    if not os.path.exists(train_manifest_path):
        raise FileNotFoundError(
            f"Train manifest not found at '{train_manifest_path}'. "
            "Please ensure data manifests are built or run data generation first."
        )

    # 2. Sanitize class indices
    sanitize_manifests(CLASSES)

    train_data = load_manifest(train_manifest_path)
    val_data = load_manifest(val_manifest_path)

    # 3. Create datasets and loaders
    train_ds = UKAccentDataset(train_data, augment=True)
    val_ds = UKAccentDataset(val_data, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        num_workers=0,  # 0 is safest on Windows with DirectML
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=0,
    )

    print(f"\n[Dataset Summary]")
    print(f"  Target Classes ({N}): {CLASSES}")
    print(f"  Train samples: {len(train_ds)} ({len(train_loader)} batches)")
    print(f"  Val samples:   {len(val_ds)} ({len(val_loader)} batches)")

    # 4. Class-weighted cross entropy
    cnt = Counter(e["class_idx"] for e in train_data)
    w = torch.tensor([1.0 / cnt.get(i, 1) for i in range(N)], dtype=torch.float32)
    w = w / w.sum() * N
    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    print("\nClass weights (inverse-frequency):")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<15}: count={cnt.get(i, 0):4d}  weight={w[i].item():.3f}")

    # 5. Initialize Model
    print(f"\nLoading WavLM Base+ model...")
    model = WavLMAccentClassifier(
        pretrained_model_name="microsoft/wavlm-base-plus",
        num_classes=N,
        freeze_encoder=True,
        unfreeze_top_k_layers=args.unfreeze_top_k,
        dropout_p=0.3,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.1f}%)\n")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-2,
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(total_steps * 0.1)),
        num_training_steps=max(total_steps, 20),
    )

    best_val_f1 = 0.0
    checkpoint_file = os.path.join(args.output_dir, "best_wavlm_accentsense.pt")

    print("=" * 65)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val F1':<10} {'Bal Acc':<10} Status")
    print("-" * 65)

    for ep in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (wav, labels, mask) in enumerate(train_loader):
            wav = wav.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            out = model(wav, attention_mask=mask)
            loss = criterion(out["logits"], labels) / args.grad_accum
            loss.backward()

            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Phase
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for wav, labels, mask in val_loader:
                wav = wav.to(device)
                mask = mask.to(device)
                out = model(wav, attention_mask=mask)
                batch_preds = out["logits"].argmax(dim=-1).cpu().tolist()
                preds.extend(batch_preds)
                targets.extend(labels.tolist())

        val_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        val_acc = balanced_accuracy_score(targets, preds)

        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, checkpoint_file)
            status = "✓ BEST"

        print(f"{ep:<6} {avg_train_loss:<12.4f} {val_f1:<10.4f} {val_acc:<10.4f} {status}")

    print("=" * 65)
    print("✓ Training Complete!")
    print(f"  Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"  Saved Checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    main()
