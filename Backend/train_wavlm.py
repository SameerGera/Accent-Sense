"""
AccentSense WavLM training.

Recommended:

    python train_wavlm.py --dry_run

then:

    python train_wavlm.py \
        --epochs 20 \
        --batch_size 8

Training strategy:

    Stage 1:
        WavLM frozen.
        Train classification head.

    Stage 2:
        Unfreeze top 2 WavLM blocks.

    Model selection:
        validation macro-F1

    Final evaluation:
        untouched test speakers
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import (
    DataLoader,
    Sampler,
)

from transformers import (
    get_cosine_schedule_with_warmup,
)


CURRENT_DIR = Path(
    __file__
).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CURRENT_DIR),
    )


from src.config import (
    CLASSES,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)

from src.data.vctk_dataset import (
    UKAccentDataset,
    collate_pad,
    load_manifest,
)

from src.models.wavlm_classifier import (
    WavLMAccentClassifier,
)


# =========================================================
# Speaker-balanced sampling
# =========================================================

class SpeakerBalancedSampler(
    Sampler
):

    """
    Sampling procedure:

        choose class uniformly
        choose speaker uniformly
        choose utterance uniformly

    Prevents prolific speakers from dominating training.
    """

    def __init__(
        self,
        manifest,
        num_samples=None,
        seed=42,
    ):

        self.seed = seed
        self.epoch = 0

        self.num_samples = int(
            num_samples
            or len(manifest)
        )

        groups = defaultdict(
            lambda:
                defaultdict(list)
        )

        for index, entry in enumerate(
            manifest
        ):

            groups[
                entry["label"]
            ][
                entry["speaker"]
            ].append(
                index
            )

        self.groups = {
            class_name: dict(
                speakers
            )

            for class_name, speakers
            in groups.items()
        }

        self.classes = [
            class_name

            for class_name in CLASSES

            if class_name
            in self.groups
        ]

        missing = [
            class_name

            for class_name in CLASSES

            if class_name
            not in self.groups
        ]

        if missing:

            raise ValueError(
                f"Training manifest "
                f"missing classes: "
                f"{missing}"
            )

    def set_epoch(
        self,
        epoch,
    ):

        self.epoch = int(
            epoch
        )

    def __len__(
        self
    ):

        return (
            self.num_samples
        )

    def __iter__(
        self
    ):

        rng = random.Random(
            self.seed
            + self.epoch
        )

        for _ in range(
            self.num_samples
        ):

            class_name = rng.choice(
                self.classes
            )

            speaker = rng.choice(
                list(
                    self.groups[
                        class_name
                    ].keys()
                )
            )

            yield rng.choice(
                self.groups[
                    class_name
                ][speaker]
            )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default=
            "microsoft/wavlm-base-plus",
    )

    parser.add_argument(
        "--manifests_dir",
        default="data/manifests",
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoints",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--head_only_epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--unfreeze_top_k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=
            SEGMENT_SECONDS,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def sanitize_manifest(
    entries
):

    output = []

    for entry in entries:

        if (
            entry.get(
                "label"
            )
            not in CLASSES
        ):
            continue

        entry = dict(
            entry
        )

        entry[
            "class_idx"
        ] = CLASSES.index(
            entry["label"]
        )

        output.append(
            entry
        )

    return output


def assert_disjoint(
    train,
    val,
    test,
):

    speaker_sets = {}

    for name, entries in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):

        speaker_sets[name] = {
            entry[
                "speaker"
            ]

            for entry in entries
        }

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):

        overlap = (
            speaker_sets[left]
            & speaker_sets[right]
        )

        if overlap:

            raise RuntimeError(
                f"Speaker leakage "
                f"{left}/{right}: "
                f"{sorted(overlap)[:10]}"
            )


def build_optimizer(
    model,
    args,
):

    head = [
        parameter

        for parameter
        in (
            list(
                model.asp.parameters()
            )
            + list(
                model.classifier.parameters()
            )
        )

        if parameter.requires_grad
    ]

    backbone = [
        parameter

        for parameter
        in model.backbone.parameters()

        if parameter.requires_grad
    ]

    groups = [
        {
            "params": head,
            "lr": args.lr_head,
        }
    ]

    if backbone:

        groups.append(
            {
                "params": backbone,
                "lr":
                    args.lr_backbone,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=
            args.weight_decay,
    )


def build_scheduler(
    optimizer,
    steps,
):

    steps = max(
        1,
        int(steps),
    )

    return (
        get_cosine_schedule_with_warmup(
            optimizer,

            num_warmup_steps=
                max(
                    1,
                    int(
                        steps
                        * 0.10
                    ),
                ),

            num_training_steps=
                steps,
        )
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
):

    model.train()

    total_loss = 0.0

    labels_all = []
    predictions_all = []

    use_amp = (
        device.type
        == "cuda"
    )

    for (
        waveform,
        labels,
        mask,
    ) in loader:

        waveform = waveform.to(
            device
        )

        labels = labels.to(
            device
        )

        mask = mask.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16,

            enabled=
                use_amp,
        ):

            logits = model(
                waveform,
                attention_mask=mask,
            )["logits"]

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        scheduler.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        labels_all.extend(
            labels
            .detach()
            .cpu()
            .tolist()
        )

        predictions_all.extend(
            logits
            .argmax(-1)
            .detach()
            .cpu()
            .tolist()
        )

    return {
        "loss":
            total_loss
            / max(
                1,
                len(labels_all),
            ),

        "acc":
            accuracy_score(
                labels_all,
                predictions_all,
            ),

        "f1":
            f1_score(
                labels_all,
                predictions_all,
                average="macro",
                zero_division=0,
            ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    return_predictions=False,
):

    model.eval()

    total_loss = 0.0

    labels_all = []
    predictions_all = []

    use_amp = (
        device.type
        == "cuda"
    )

    for (
        waveform,
        labels,
        mask,
    ) in loader:

        waveform = waveform.to(
            device
        )

        labels = labels.to(
            device
        )

        mask = mask.to(
            device
        )

        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16,

            enabled=
                use_amp,
        ):

            logits = model(
                waveform,
                attention_mask=mask,
            )["logits"]

            loss = criterion(
                logits,
                labels,
            )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        labels_all.extend(
            labels.cpu().tolist()
        )

        predictions_all.extend(
            logits
            .argmax(-1)
            .cpu()
            .tolist()
        )

    metrics = {

        "loss":
            total_loss
            / max(
                1,
                len(labels_all),
            ),

        "acc":
            accuracy_score(
                labels_all,
                predictions_all,
            ),

        "bal_acc":
            balanced_accuracy_score(
                labels_all,
                predictions_all,
            ),

        "f1":
            f1_score(
                labels_all,
                predictions_all,
                average="macro",
                zero_division=0,
            ),
    }

    if return_predictions:

        return (
            metrics,
            labels_all,
            predictions_all,
        )

    return metrics


def save_checkpoint(
    path,
    model,
    epoch,
    val_f1,
    args,
):

    torch.save(
        {
            "model_state":
                model.state_dict(),

            "classes":
                CLASSES,

            "sample_rate":
                SAMPLE_RATE,

            "segment_seconds":
                args.segment_seconds,

            "model_name":
                args.model_name,

            "epoch":
                epoch,

            "val_f1":
                val_f1,

            "unfreeze_top_k":
                args.unfreeze_top_k,
        },

        path,
    )


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if (
        device.type != "cuda"
        and not args.dry_run
    ):

        print(
            "WARNING: CUDA unavailable. "
            "Training will be slow."
        )

    manifest_dir = Path(
        args.manifests_dir
    )

    train = sanitize_manifest(
        load_manifest(
            manifest_dir
            / "train.json"
        )
    )

    val = sanitize_manifest(
        load_manifest(
            manifest_dir
            / "val.json"
        )
    )

    test = sanitize_manifest(
        load_manifest(
            manifest_dir
            / "test.json"
        )
    )

    if (
        not train
        or not val
        or not test
    ):

        raise RuntimeError(
            "Train/val/test must "
            "all be non-empty."
        )

    assert_disjoint(
        train,
        val,
        test,
    )

    print(
        f"Device: {device}"
    )

    print(
        "Train / Val / Test clips: "
        f"{len(train)} / "
        f"{len(val)} / "
        f"{len(test)}"
    )

    print(
        "Train speakers:",
        len(
            {
                entry["speaker"]
                for entry in train
            }
        ),
    )

    print(
        "Val speakers:",
        len(
            {
                entry["speaker"]
                for entry in val
            }
        ),
    )

    print(
        "Test speakers:",
        len(
            {
                entry["speaker"]
                for entry in test
            }
        ),
    )

    train_dataset = UKAccentDataset(
        train,
        augment=True,
        segment_seconds=
            args.segment_seconds,
    )

    val_dataset = UKAccentDataset(
        val,
        augment=False,
        segment_seconds=
            args.segment_seconds,
    )

    test_dataset = UKAccentDataset(
        test,
        augment=False,
        segment_seconds=
            args.segment_seconds,
    )

    sampler = (
        SpeakerBalancedSampler(
            train,
            seed=args.seed,
        )
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=
            args.batch_size,

        sampler=
            sampler,

        collate_fn=
            collate_pad,

        num_workers=
            args.num_workers,

        pin_memory=
            (
                device.type
                == "cuda"
            ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        collate_fn=
            collate_pad,

        num_workers=
            args.num_workers,

        pin_memory=
            (
                device.type
                == "cuda"
            ),
    )

    test_loader = DataLoader(
        test_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        collate_fn=
            collate_pad,

        num_workers=
            args.num_workers,

        pin_memory=
            (
                device.type
                == "cuda"
            ),
    )

    model = (
        WavLMAccentClassifier(

            pretrained_model_name=
                args.model_name,

            num_classes=
                len(CLASSES),

            freeze_encoder=True,

            unfreeze_top_k_layers=0,
        )
        .to(
            device
        )
    )

    trainable, total = (
        model
        .trainable_parameter_counts()
    )

    print(
        "Initial trainable parameters: "
        f"{trainable:,}/{total:,}"
    )

    if args.dry_run:

        (
            waveform,
            labels,
            mask,
        ) = next(
            iter(
                train_loader
            )
        )

        with torch.no_grad():

            output = model(
                waveform.to(
                    device
                ),

                attention_mask=
                    mask.to(
                        device
                    ),
            )

        print(
            "Dry-run logits:",
            tuple(
                output[
                    "logits"
                ].shape
            ),
        )

        print(
            "SUCCESS: dataset + "
            "WavLM forward pass works."
        )

        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "best_wavlm_accentsense.pt"
    )

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=
                args.label_smoothing
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=(
                device.type
                == "cuda"
            ),
        )
    )

    # =====================================================
    # Stage 1
    # =====================================================

    model.freeze_backbone()

    optimizer = build_optimizer(
        model,
        args,
    )

    head_epochs = min(
        args.head_only_epochs,
        args.epochs,
    )

    scheduler = build_scheduler(
        optimizer,

        len(train_loader)
        * max(
            1,
            head_epochs,
        ),
    )

    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ================================================
        # Stage 2
        # ================================================

        if (
            epoch
            == head_epochs + 1
            and epoch <= args.epochs
        ):

            print(
                "\n[Stage 2] "
                f"Unfreezing top "
                f"{args.unfreeze_top_k} "
                "WavLM blocks"
            )

            model.unfreeze_top_k(
                args.unfreeze_top_k
            )

            trainable, total = (
                model
                .trainable_parameter_counts()
            )

            print(
                "Trainable parameters: "
                f"{trainable:,}/{total:,}"
            )

            optimizer = build_optimizer(
                model,
                args,
            )

            remaining_epochs = (
                args.epochs
                - head_epochs
            )

            scheduler = (
                build_scheduler(
                    optimizer,

                    len(train_loader)
                    * max(
                        1,
                        remaining_epochs,
                    ),
                )
            )

            bad_epochs = 0

        sampler.set_epoch(
            epoch
        )

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                scaler,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{args.epochs:02d} | "

            f"Train loss "
            f"{train_metrics['loss']:.4f} "

            f"acc "
            f"{train_metrics['acc']:.3f} "

            f"F1 "
            f"{train_metrics['f1']:.3f} | "

            f"Val loss "
            f"{val_metrics['loss']:.4f} "

            f"acc "
            f"{val_metrics['acc']:.3f} "

            f"bal "
            f"{val_metrics['bal_acc']:.3f} "

            f"F1 "
            f"{val_metrics['f1']:.3f}"
        )

        if (
            val_metrics["f1"]
            > best_f1 + 1e-4
        ):

            best_f1 = (
                val_metrics[
                    "f1"
                ]
            )

            bad_epochs = 0

            save_checkpoint(
                best_path,
                model,
                epoch,
                best_f1,
                args,
            )

            print(
                "  -> saved best "
                "checkpoint "
                f"(macro-F1="
                f"{best_f1:.4f})"
            )

        else:

            bad_epochs += 1

        if (
            epoch > head_epochs
            and bad_epochs
            >= args.patience
        ):

            print(
                "Early stopping: "
                f"{args.patience} "
                "epochs without "
                "validation-F1 improvement."
            )

            break

    # =====================================================
    # Final held-out test
    # =====================================================

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    print(
        "\nReloaded best epoch "
        f"{checkpoint['epoch']} "
        f"(val F1="
        f"{checkpoint['val_f1']:.4f})"
    )

    (
        test_metrics,
        true_labels,
        predictions,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        return_predictions=True,
    )

    print(
        "TEST | "
        f"loss "
        f"{test_metrics['loss']:.4f} "

        f"acc "
        f"{test_metrics['acc']:.3f} "

        f"bal "
        f"{test_metrics['bal_acc']:.3f} "

        f"macro-F1 "
        f"{test_metrics['f1']:.3f}"
    )

    report = classification_report(
        true_labels,
        predictions,

        labels=
            list(
                range(
                    len(CLASSES)
                )
            ),

        target_names=
            list(CLASSES),

        output_dict=True,

        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,

        labels=
            list(
                range(
                    len(CLASSES)
                )
            ),
    ).tolist()

    report_path = (
        output_dir
        / "test_metrics.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "best_epoch":
                    checkpoint[
                        "epoch"
                    ],

                "best_val_f1":
                    checkpoint[
                        "val_f1"
                    ],

                "test":
                    test_metrics,

                "classification_report":
                    report,

                "confusion_matrix":
                    matrix,

                "classes":
                    CLASSES,
            },

            indent=2,
        ),

        encoding="utf-8",
    )

    print(
        f"Test report: "
        f"{report_path}"
    )

    print(
        f"Best model: "
        f"{best_path}"
    )


if __name__ == "__main__":
    main()