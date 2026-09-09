"""
Training and Evaluation Script for Phase 1 Classical Acoustic Baseline.
Run with: python train_baseline.py [--data_csv PATH]
"""

import argparse
import os
import pandas as pd
import numpy as np
from src.data.svarah_dataset import build_speaker_disjoint_splits
from src.models.baseline_mfcc import ClassicalAcousticBaseline, extract_acoustic_features
from tqdm import tqdm


def run_baseline_experiment(metadata_path: str = None, synthetic_fallback: bool = True):
    print("=" * 60)
    print("AccentSense: Classical Acoustic Baseline (MFCC + SVM/RF)")
    print("=" * 60)

    if metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"Loaded dataset metadata: {len(df)} rows.")
    else:
        if synthetic_fallback:
            print("[INFO] No external metadata CSV provided. Generating synthetic multi-speaker benchmark for verification...")
            np.random.seed(42)
            n_samples = 300
            speakers = [f"spk_{i}" for i in range(1, 31)]  # 30 speakers
            languages = ["Northern_Hindi", "Central_MP", "Western_Gujarati", "Southern_Tamil"]

            rows = []
            for i in range(n_samples):
                spk = np.random.choice(speakers)
                # Ensure speaker is primarily tied to one language to mimic reality
                spk_idx = int(spk.split("_")[1])
                lang = languages[spk_idx % len(languages)]
                rows.append({
                    "speaker_id": spk,
                    "primary_language": lang,
                    "dummy_audio_feature": np.random.randn(80),
                })
            df = pd.DataFrame(rows)
        else:
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    # Build speaker-disjoint splits
    train_df, val_df, test_df = build_speaker_disjoint_splits(
        metadata_df=df,
        target_col="primary_language",
        speaker_col="speaker_id",
        n_splits=5,
    )

    # Feature preparation
    target_names = sorted(train_df["target"].unique())
    label_to_id = {name: i for i, name in enumerate(target_names)}

    print(f"Target classes ({len(target_names)}): {target_names}")

    if "dummy_audio_feature" in df.columns:
        X_train = np.stack(train_df["dummy_audio_feature"].values)
        X_test = np.stack(test_df["dummy_audio_feature"].values)
    else:
        print("Extracting acoustic features from audio files...")
        X_train = np.array([extract_acoustic_features(p) for p in tqdm(train_df["audio_path"])])
        X_test = np.array([extract_acoustic_features(p) for p in tqdm(test_df["audio_path"])])

    y_train = np.array([label_to_id[lbl] for lbl in train_df["target"]])
    y_test = np.array([label_to_id[lbl] for lbl in test_df["target"]])

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
    parser.add_argument("--data_csv", type=str, default=None, help="Path to Svarah metadata CSV")
    args = parser.parse_args()
    run_baseline_experiment(metadata_path=args.data_csv)
