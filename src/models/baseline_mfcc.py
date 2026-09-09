"""
Phase 1 Baseline: Acoustic MFCC + Prosodic Feature Extraction + SVM / Random Forest.
"""

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from typing import Tuple, Dict, Any, List
import joblib


def extract_acoustic_features(audio_path_or_waveform, sr: int = 16000) -> np.ndarray:
    """
    Extracts 13-dim MFCCs + delta + delta-delta + Pitch (F0) + RMS energy,
    summarized with mean, std, skew across time.
    Total feature vector length: ~80-120 dimensions.
    """
    if isinstance(audio_path_or_waveform, str):
        y, orig_sr = librosa.load(audio_path_or_waveform, sr=sr)
    else:
        y = np.asarray(audio_path_or_waveform, dtype=np.float32)

    # 1. MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # 2. Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # 3. Energy / RMS
    rms = librosa.feature.rms(y=y)

    # 4. Fundamental Frequency (F0 / Pitch estimation via Yin)
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0_clean = np.nan_to_num(f0, nan=0.0)

    # Concatenate all frame-level features: shape [num_features, time_frames]
    all_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, centroid, rolloff, rms, f0_clean.reshape(1, -1)])

    # Aggregate across time dimension (mean, std)
    mean_feat = np.mean(all_features, axis=1)
    std_feat = np.std(all_features, axis=1)
    max_feat = np.max(all_features, axis=1)
    min_feat = np.min(all_features, axis=1)

    feature_vector = np.concatenate([mean_feat, std_feat, max_feat, min_feat])
    return feature_vector


class ClassicalAcousticBaseline:
    """
    Classical acoustic feature baseline using SVM with RBF kernel and class weighting.
    """
    def __init__(self, classifier_type: str = "svm_rbf", random_state: int = 42):
        self.classifier_type = classifier_type
        if classifier_type == "svm_rbf":
            self.model = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=random_state)
        elif classifier_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)
        else:
            self.model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, target_names: List[str] = None) -> Dict[str, Any]:
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=target_names, zero_division=0)

        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def save(self, file_path: str):
        joblib.dump(self.model, file_path)

    def load(self, file_path: str):
        self.model = joblib.load(file_path)
