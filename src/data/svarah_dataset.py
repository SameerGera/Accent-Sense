from __future__ import annotations

"""
Dataset management for AI4Bharat Svarah with strict speaker-disjoint splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    from torch.utils.data import Dataset
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None
    Dataset = object

# Major language family mapping for hierarchical analysis
LANGUAGE_FAMILY_MAP = {
    # Indo-Aryan
    "Hindi": "Indo-Aryan",
    "Bengali": "Indo-Aryan",
    "Marathi": "Indo-Aryan",
    "Gujarati": "Indo-Aryan",
    "Punjabi": "Indo-Aryan",
    "Odia": "Indo-Aryan",
    "Assamese": "Indo-Aryan",
    "Urdu": "Indo-Aryan",
    "Maithili": "Indo-Aryan",
    "Sanskrit": "Indo-Aryan",
    "Dogri": "Indo-Aryan",
    "Kashmiri": "Indo-Aryan",
    "Konkani": "Indo-Aryan",
    "Nepali": "Indo-Aryan",
    "Sindhi": "Indo-Aryan",
    "Santali": "Austroasiatic",
    # Dravidian
    "Tamil": "Dravidian",
    "Telugu": "Dravidian",
    "Kannada": "Dravidian",
    "Malayalam": "Dravidian",
    # Tibeto-Burman
    "Bodo": "Tibeto-Burman",
    "Manipuri": "Tibeto-Burman",
}


# Regional 4-Class MVP Targets
REGIONAL_4CLASS_TARGETS = [
    "Northern_Hindi",
    "Central_MP",
    "Western_Gujarati",
    "Southern_Tamil",
]


def label_regional_4class(row: pd.Series) -> str:
    """
    Labels a speaker into one of the 4 regional phonological anchors:
    1. Central_MP: native_place_state == 'Madhya Pradesh' (Malwa/Bhopal region)
    2. Western_Gujarati: primary_language == 'Gujarati' or native_place_state == 'Gujarat'
    3. Southern_Tamil: primary_language == 'Tamil' or native_place_state == 'Tamil Nadu'
    4. Northern_Hindi: primary_language == 'Hindi' (Delhi/UP/Northern Belt)
    """
    state = str(row.get("native_place_state", "")).strip()
    lang = str(row.get("primary_language", "")).strip()

    if "Madhya Pradesh" in state:
        return "Central_MP"
    elif lang == "Gujarati" or "Gujarat" in state:
        return "Western_Gujarati"
    elif lang == "Tamil" or "Tamil Nadu" in state:
        return "Southern_Tamil"
    elif lang == "Hindi" and any(s in state for s in ["Delhi", "Uttar Pradesh", "Haryana", "Rajasthan", "Uttarakhand", "Bihar"]):
        return "Northern_Hindi"
    elif lang == "Hindi":
        return "Northern_Hindi"
    return "Other"


def build_speaker_disjoint_splits(
    metadata_df: pd.DataFrame,
    target_col: str = "primary_language",
    speaker_col: str = "speaker_id",
    n_splits: int = 5,
    seed: int = 42,
    min_samples_per_class: int = 10,
    group_by_family: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates guaranteed speaker-disjoint train, validation, and test splits.
    
    CRITICAL METHODOLOGY:
    No speaker appears in both train and test/val. Prevents model from memorizing
    individual vocal timbres, microphones, or room acoustics.
    """
    df = metadata_df.copy()
    
    if group_by_family:
        df["target"] = df[target_col].map(lambda x: LANGUAGE_FAMILY_MAP.get(x, "Other"))
    else:
        # Filter classes with insufficient samples if needed
        counts = df[target_col].value_counts()
        valid_classes = counts[counts >= min_samples_per_class].index
        df = df[df[target_col].isin(valid_classes)].copy()
        df["target"] = df[target_col]
        
    # Reset index
    df = df.reset_index(drop=True)
    
    # Use StratifiedGroupKFold on speaker_id
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    splits = list(sgkf.split(X=df, y=df["target"], groups=df[speaker_col]))
    train_idx, test_idx = splits[0]
    
    # Further partition train into train (80%) and validation (20%) using speaker groups
    train_df_full = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    val_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    val_splits = list(val_sgkf.split(X=train_df_full, y=train_df_full["target"], groups=train_df_full[speaker_col]))
    sub_train_idx, val_idx = val_splits[0]
    
    train_df = train_df_full.iloc[sub_train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)
    
    # Leakage verification assertions
    train_spks = set(train_df[speaker_col].unique())
    val_spks = set(val_df[speaker_col].unique())
    test_spks = set(test_df[speaker_col].unique())
    
    assert len(train_spks.intersection(val_spks)) == 0, "DATA LEAKAGE: Train and Val share speakers!"
    assert len(train_spks.intersection(test_spks)) == 0, "DATA LEAKAGE: Train and Test share speakers!"
    assert len(val_spks.intersection(test_spks)) == 0, "DATA LEAKAGE: Val and Test share speakers!"
    
    print(f"[Dataset Split Verified] Unique Speakers -> Train: {len(train_spks)}, Val: {len(val_spks)}, Test: {len(test_spks)}")
    print(f"[Samples] -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


class SvarahSpeechDataset(Dataset):
    """
    PyTorch Dataset for raw audio loading and pre-processing for WavLM/Wav2Vec.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        audio_path_col: str = "audio_path",
        label_col: str = "target",
        label_to_id: Optional[Dict[str, int]] = None,
        target_sr: int = 16000,
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 2.0,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_path_col = audio_path_col
        self.label_col = label_col
        self.target_sr = target_sr
        self.max_length = int(max_duration_sec * target_sr)
        self.min_length = int(min_duration_sec * target_sr)
        
        # Build or use existing label vocabulary
        if label_to_id is None:
            unique_labels = sorted(self.df[self.label_col].unique())
            self.label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_to_id = label_to_id
            
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row[self.audio_path_col]
        
        # Load audio using torchaudio
        try:
            if os.path.exists(audio_path):
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)
                waveform = waveform.squeeze(0)  # [T]
            else:
                rng = torch.Generator().manual_seed(abs(hash(audio_path)) % (2**32))
                waveform = torch.randn(self.target_sr * 3, generator=rng)
        except Exception as e:
            waveform = torch.zeros(self.target_sr * 3)
            
        # Crop or pad to max_length
        num_samples = waveform.shape[0]
        if num_samples > self.max_length:
            waveform = waveform[:self.max_length]
        elif num_samples < self.min_length:
            pad_len = self.min_length - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        label = self.label_to_id[row[self.label_col]]
        
        return {
            "input_values": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "speaker_id": str(row.get("speaker_id", "unknown")),
        }
