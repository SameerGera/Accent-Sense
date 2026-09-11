"""
VCTK + Mozilla Common Voice Dataset Loader for UK Regional Accent Classification.

Supports two data sources:
  1. CSTR VCTK Corpus   — 109 speakers, multiple UK accents, 44kHz -> resample 16kHz
  2. Mozilla Common Voice (en) — used for Irish English and Cockney supplementation

7 Target Classes: RP, Scottish, Welsh, Northern, West_Midlands, Cockney, Irish
"""

import os
import json
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Class Definitions
# ---------------------------------------------------------------------------
CLASSES = ["RP", "Scottish", "Welsh", "Northern", "West_Midlands", "Cockney", "Irish"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# VCTK accent tag -> our class mapping
# (based on speaker-info.txt ACCENT column in VCTK corpus)
# ---------------------------------------------------------------------------
VCTK_ACCENT_MAP: Dict[str, str] = {
    # RP / Standard Southern British
    "English":          "RP",
    "Southern England": "RP",
    "Surrey":           "RP",
    "Kent":             "RP",
    "Hampshire":        "RP",
    "Essex":            "RP",
    "Berkshire":        "RP",
    "Oxfordshire":      "RP",
    # Scottish
    "Scottish":         "Scottish",
    "Scotland":         "Scottish",
    "Edinburgh":        "Scottish",
    "Glasgow":          "Scottish",
    # Welsh
    "Welsh":            "Welsh",
    "Wales":            "Welsh",
    # Northern English
    "Northern England": "Northern",
    "Yorkshire":        "Northern",
    "Manchester":       "Northern",
    "Newcastle":        "Northern",
    "Geordie":          "Northern",
    "Lancashire":       "Northern",
    "Leeds":            "Northern",
    # West Midlands
    "West Midlands":    "West_Midlands",
    "Birmingham":       "West_Midlands",
    "Midlands":         "West_Midlands",
    # Cockney / London
    "London":           "Cockney",
    "Cockney":          "Cockney",
    "East London":      "Cockney",
    # Irish — primarily from Common Voice
    "Irish":            "Irish",
    "Ireland":          "Irish",
    "Dublin":           "Irish",
    "Northern Ireland": "Irish",
}

TARGET_SAMPLE_RATE = 16_000


def parse_vctk_speaker_info(speaker_info_path: str) -> Dict[str, str]:
    """
    Parses VCTK speaker-info.txt and returns {speaker_id: accent_class}.
    Lines look like: p225 | 23 | F | Southern England | English
    """
    speaker_to_class: Dict[str, str] = {}
    with open(speaker_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ID"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            speaker_id = parts[0]          # e.g. "p225"
            accent_tag = parts[4]          # e.g. "English"
            cls = VCTK_ACCENT_MAP.get(accent_tag)
            if cls is None:
                # Try partial match
                for key, val in VCTK_ACCENT_MAP.items():
                    if key.lower() in accent_tag.lower():
                        cls = val
                        break
            if cls is not None:
                speaker_to_class[speaker_id] = cls
    return speaker_to_class


def build_vctk_manifest(
    vctk_wav_dir: str,
    speaker_info_path: str,
) -> List[Dict]:
    """
    Scans VCTK wav directory and builds a flat list of
    {"path": ..., "label": ..., "speaker": ..., "class_idx": ...}
    """
    speaker_to_class = parse_vctk_speaker_info(speaker_info_path)
    wav_dir = Path(vctk_wav_dir)
    manifest = []

    for wav_path in sorted(wav_dir.rglob("*.wav")):
        # VCTK naming: wav48/p225/p225_001.wav  or  wav16/p225/p225_001.wav
        speaker_id = wav_path.parent.name
        if speaker_id not in speaker_to_class:
            # Try stripping mic suffix: p225_mic1 -> p225
            base = speaker_id.split("_")[0]
            if base not in speaker_to_class:
                continue
            speaker_id = base
        cls = speaker_to_class[speaker_id]
        manifest.append({
            "path": str(wav_path),
            "label": cls,
            "speaker": speaker_id,
            "class_idx": CLASS_TO_IDX[cls],
        })

    return manifest


def build_cv_manifest(
    cv_clips_dir: str,
    tsv_path: str,
    target_class: str = "Irish",
) -> List[Dict]:
    """
    Builds a manifest from Mozilla Common Voice validated.tsv for a given target class.
    Expects TSV with columns: client_id, path, sentence, ...
    """
    manifest = []
    clips_dir = Path(cv_clips_dir)
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        path_col = header.index("path") if "path" in header else 1
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= path_col:
                continue
            rel_path = parts[path_col]
            wav_path = clips_dir / rel_path
            if not wav_path.exists():
                # Try with .mp3 extension if no extension present
                if not rel_path.endswith((".wav", ".mp3")):
                    wav_path = clips_dir / (rel_path + ".mp3")
            if wav_path.exists():
                manifest.append({
                    "path": str(wav_path),
                    "label": target_class,
                    "speaker": parts[0] if parts else "unknown",
                    "class_idx": CLASS_TO_IDX[target_class],
                })
    return manifest


def speaker_disjoint_splits(
    manifest: List[Dict],
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits manifest into train/val/test with no speaker overlap across splits.
    Stratifies by class to keep class balance consistent.
    """
    random.seed(seed)

    # Group speakers by class
    class_to_speakers: Dict[str, List[str]] = {}
    for entry in manifest:
        cls = entry["label"]
        spk = entry["speaker"]
        class_to_speakers.setdefault(cls, [])
        if spk not in class_to_speakers[cls]:
            class_to_speakers[cls].append(spk)

    train_speakers, val_speakers, test_speakers = set(), set(), set()

    for cls, speakers in class_to_speakers.items():
        random.shuffle(speakers)
        n = len(speakers)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_speakers.update(speakers[:n_train])
        val_speakers.update(speakers[n_train:n_train + n_val])
        test_speakers.update(speakers[n_train + n_val:])

    train = [e for e in manifest if e["speaker"] in train_speakers]
    val   = [e for e in manifest if e["speaker"] in val_speakers]
    test  = [e for e in manifest if e["speaker"] in test_speakers]

    return train, val, test


class UKAccentDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for UK regional accent classification.
    Works with manifest lists produced by build_vctk_manifest / build_cv_manifest.

    Returns (waveform_tensor [T], class_idx) where waveform is mono 16kHz, max 10s.
    """
    MAX_SAMPLES = TARGET_SAMPLE_RATE * 10  # 10 seconds

    def __init__(
        self,
        manifest: List[Dict],
        augment: bool = False,
    ):
        self.manifest = manifest
        self.augment = augment
        self.resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16kHz
        if sr != TARGET_SAMPLE_RATE:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
            waveform = self.resampler_cache[sr](waveform)
        # Truncate to max length
        if waveform.shape[1] > self.MAX_SAMPLES:
            start = random.randint(0, waveform.shape[1] - self.MAX_SAMPLES)
            waveform = waveform[:, start:start + self.MAX_SAMPLES]
        return waveform.squeeze(0)  # [T]

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Speed perturbation ±5%
        if random.random() < 0.5:
            rate = random.uniform(0.95, 1.05)
            effects = [["rate", str(int(TARGET_SAMPLE_RATE * rate))]]
            try:
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform.unsqueeze(0), TARGET_SAMPLE_RATE, effects
                )
                waveform = waveform.squeeze(0)
            except Exception:
                pass
        # Additive Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.manifest[idx]
        try:
            waveform = self._load_audio(entry["path"])
        except Exception as e:
            print(f"[UKAccentDataset] Error loading {entry['path']}: {e}")
            waveform = torch.zeros(TARGET_SAMPLE_RATE * 3)  # 3s silence fallback
        if self.augment:
            waveform = self._augment(waveform)
        return waveform, entry["class_idx"]


def collate_pad(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads variable-length waveforms in a batch to the maximum length.
    """
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        for w in waveforms
    ])
    return padded, torch.tensor(labels, dtype=torch.long)


def save_manifest(manifest: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[UKAccentDataset] Saved manifest ({len(manifest)} items) -> {path}")


def load_manifest(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
