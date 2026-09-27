"""
Training and Fine-Tuning Pipeline for WavLM Base+ on UK Regional Accent Classification.
Ready for Local GPU execution (CUDA / DirectML) or Google Colab.

Usage:
    python train_wavlm.py --epochs 20 --batch_size 8 --lr 3e-4
    python train_wavlm.py --dry_run  # sanity check without training
"""

import argparse
import os
import sys
import json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Ensure Backend directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data.vctk_dataset import UKAccentDataset, collate_pad, load_manifest
from src.models.wavlm_classifier import WavLMAccentClassifier

# 6 UK Regional Accent Classes
CLASSES = ("RP", "Scottish", "Welsh", "Northern", "West_Midlands", "Irish")
N = len(CLASSES)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for wav, labels, mask in dataloader:
        wav = wav.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(wav, attention_mask=mask)
        logits = outputs["logits"]

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

    epoch_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, macro_f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for wav, labels, mask in dataloader:
            wav = wav.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(wav, attention_mask=mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, bal_acc, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train WavLM Base+ for UK Regional Accent Classification.")
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-base-plus")
    parser.add_argument("--manifests_dir", type=str, default="data/manifests", help="Directory containing JSON manifests")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze_top_k", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Run single batch sanity check")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training but was not detected.\n"
            "Please ensure you have an NVIDIA GPU with CUDA drivers installed.\n"
            "For Google Colab: Runtime -> Change runtime type -> T4 GPU"
        )
    device = torch.device("cuda")
    print("=" * 65)
    print("AccentSense: WavLM Base+ UK Accent Classification Training")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Model: {args.model_name}")
    print(f"Classes ({N}): {CLASSES}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load JSON manifests
    train_path = os.path.join(args.manifests_dir, "train.json")
    val_path = os.path.join(args.manifests_dir, "val.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train manifest not found at `{train_path}`. "
            "Run the notebook or curate_data.py first, or use --manifests_dir."
        )

    train_manifest = load_manifest(train_path)
    val_manifest = load_manifest(val_path) if os.path.exists(val_path) else []

    # 2. Sanitize class indices
    for split_name, split_data in [("train", train_manifest), ("val", val_manifest)]:
        for e in split_data:
            if e["label"] in CLASSES:
                e["class_idx"] = CLASSES.index(e["label"])

    print(f"Loaded Splits -> Train: {len(train_manifest)} samples | Val: {len(val_manifest)} samples")

    # 3. Build Datasets & DataLoaders
    train_dataset = UKAccentDataset(train_manifest, augment=True)
    val_dataset = UKAccentDataset(val_manifest, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=2,
    )

    # 4. Compute Class Weights for balanced loss
    class_counts = Counter(e["class_idx"] for e in train_manifest)
    total_samples = len(train_manifest)
    weights = [total_samples / (N * class_counts.get(i, 1)) for i in range(N)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.dry_run:
        print("\n[Dry Run Sanity Check]")
        print(f"Class Weights: {weights}")
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        print(f"[SUCCESS] Ready for GPU training on {device}.")
        return

    # 5. Initialize WavLM Model
    print(f"\n[Model Initialization] Loading {args.model_name}...")
    model = WavLMAccentClassifier(
        pretrained_model_name=args.model_name,
        num_classes=N,
        freeze_encoder=args.freeze_encoder,
        unfreeze_top_k_layers=args.unfreeze_top_k,
    ).to(device)

    # Separate parameter groups for backbone vs head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.asp.parameters()] + [p for p in model.classifier.parameters()]

    optimizer_grouped_parameters = [
        {"params": head_params, "lr": args.lr},
    ]
    if len(backbone_params) > 0:
        optimizer_grouped_parameters.append({"params": backbone_params, "lr": args.lr / 10})

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # 6. Training Loop
    best_val_f1 = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_wavlm_accentsense.pt")

    print(f"\n[Training Kickoff] Running for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc, val_bal_acc, val_f1 = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} BalAcc: {val_bal_acc:.3f} F1: {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to: {best_checkpoint_path} (Val Macro-F1: {val_f1:.4f})")

    print("\n[Training Complete] Model training loop executed successfully.")
    print(f"  Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
