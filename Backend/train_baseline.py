"""
Training and Evaluation Script for Phase 1 Classical Acoustic Baseline.
6 UK Regional Accent Classes: RP, Scottish, Welsh, Northern, West_Midlands, Irish

Usage:
    python train_baseline.py [--data_csv PATH] [--splits_dir DIR]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data.svarah_dataset import build_speaker_disjoint_splits, REGIONAL_6CLASS_TARGETS
from src.models.baseline_mfcc import ClassicalAcousticBaseline, extract_acoustic_features
from tqdm import tqdm


def run_baseline_experiment(metadata_path: str = None, splits_dir: str = None, synthetic_fallback: bool = True):
    print("=" * 60)
    print("AccentSense: Classical Acoustic Baseline (MFCC + SVM/RF)")
    print("6 UK Regional Accent Classes")
    print("=" * 60)

    # 1. Check if curated Phase 2 splits are available
    check_dir = splits_dir or "data/splits"
    train_split_path = os.path.join(check_dir, "train_speaker_disjoint.csv")
    test_split_path = os.path.join(check_dir, "test_speaker_disjoint.csv")

    if os.path.exists(train_split_path) and os.path.exists(test_split_path):
        print(f"[Phase 2 Integration] Loading curated speaker-disjoint splits from: `{check_dir}/`")
        train_df = pd.read_csv(train_split_path)
        test_df = pd.read_csv(test_split_path)
        print(f"Loaded Curated Splits -> Train: {len(train_df)} samples ({train_df['speaker_id'].nunique()} speakers) | Test: {len(test_df)} samples ({test_df['speaker_id'].nunique()} speakers)")
    elif metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"Loaded dataset metadata: {len(df)} rows.")
        train_df, val_df, test_df = build_speaker_disjoint_splits(
            metadata_df=df,
            target_col="accent",
            speaker_col="speaker_id",
            n_splits=5,
        )
    else:
        if synthetic_fallback:
            print("[INFO] No external metadata CSV provided. Generating synthetic multi-speaker benchmark for verification...")
            np.random.seed(42)
            n_samples = 300
            speakers = [f"spk_{i}" for i in range(1, 31)]  # 30 speakers
            accents = REGIONAL_6CLASS_TARGETS

            rows = []
            for i in range(n_samples):
                spk = np.random.choice(speakers)
                spk_idx = int(spk.split("_")[1])
                accent = accents[spk_idx % len(accents)]
                rows.append({
                    "speaker_id": spk,
                    "accent": accent,
                    "dummy_audio_feature": np.random.randn(80),
                })
            df = pd.DataFrame(rows)
            train_df, val_df, test_df = build_speaker_disjoint_splits(
                metadata_df=df,
                target_col="accent",
                speaker_col="speaker_id",
                n_splits=5,
            )
        else:
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    # Feature preparation
    target_names = sorted(train_df["target"].unique()) if "target" in train_df.columns else sorted(train_df["accent"].unique())
    label_to_id = {name: i for i, name in enumerate(target_names)}

    print(f"Target classes ({len(target_names)}): {target_names}")

    # Use dummy features if available, otherwise extract from audio
    if "dummy_audio_feature" in train_df.columns:
        X_train = np.stack(train_df["dummy_audio_feature"].values)
        X_test = np.stack(test_df["dummy_audio_feature"].values)
    elif "audio_path" in train_df.columns:
        print("Extracting acoustic features from audio files...")
        X_train = np.array([extract_acoustic_features(p) for p in tqdm(train_df["audio_path"])])
        X_test = np.array([extract_acoustic_features(p) for p in tqdm(test_df["audio_path"])])
    else:
        print("[WARNING] No audio data available. Using random features for pipeline validation.")
        X_train = np.random.randn(len(train_df), 80)
        X_test = np.random.randn(len(test_df), 80)

    y_train = np.array([label_to_id[lbl] for lbl in (train_df["target"] if "target" in train_df.columns else train_df["accent"])])
    y_test = np.array([label_to_id[lbl] for lbl in (test_df["target"] if "target" in test_df.columns else test_df["accent"])])

    # Train SVM baseline
    print("\n--- Training SVM (RBF Kernel, Balanced) ---")
    svm_baseline = ClassicalAcousticBaseline(classifier_type="svm_rbf")
    svm_baseline.fit(X_train, y_train)
    results = svm_baseline.evaluate(X_test, y_test, target_names=target_names)

    print("\n[Baseline Results - Test Set (Speaker-Disjoint)]")
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {results['macro_f1']:.4f}")
    print(f"Weighted-F1:       {results['weighted_f1']:.4f}")
    print("\nClassification Report:\n", results["classification_report"])

    # Train Random Forest baseline
    print("\n--- Training Random Forest Baseline ---")
    rf_baseline = ClassicalAcousticBaseline(classifier_type="random_forest")
    rf_baseline.fit(X_train, y_train)
    rf_results = rf_baseline.evaluate(X_test, y_test, target_names=target_names)
    print(f"RF Macro-F1: {rf_results['macro_f1']:.4f} | Balanced Accuracy: {rf_results['balanced_accuracy']:.4f}")

    print("\n[Phase 1 Baseline Complete] Ready for comparative benchmarking against WavLM.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Path to curated splits directory")
    args = parser.parse_args()
    run_baseline_experiment(metadata_path=args.data_csv, splits_dir=args.splits_dir)
