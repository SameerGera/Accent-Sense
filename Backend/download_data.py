"""
AccentSense Data Download & Manifest Builder
Downloads UK accent audio data from Hugging Face and builds speaker-disjoint train/val/test splits.

Usage:
    huggingface-cli login
    cd Backend
    python download_data.py

Output:
    Backend/data/manifests/train.json
    Backend/data/manifests/val.json
    Backend/data/manifests/test.json
"""

import os
import sys
import json
import random
import subprocess
from collections import Counter

import numpy as np
import soundfile as sf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 6 UK Regional Accent Classes
CLASSES = ("RP", "Scottish", "Welsh", "Northern", "West_Midlands", "Irish")

VCTK_ACCENT_MAP = {
    "English": "RP", "Southern England": "RP", "Surrey": "RP", "Kent": "RP",
    "Hampshire": "RP", "Essex": "RP", "Berkshire": "RP", "Oxfordshire": "RP",
    "Scottish": "Scottish", "Scotland": "Scottish", "Edinburgh": "Scottish", "Glasgow": "Scottish",
    "Welsh": "Welsh", "Wales": "Welsh",
    "Northern England": "Northern", "Yorkshire": "Northern", "Manchester": "Northern",
    "Newcastle": "Northern", "Geordie": "Northern", "Lancashire": "Northern", "Leeds": "Northern",
    "West Midlands": "West_Midlands", "Birmingham": "West_Midlands", "Midlands": "West_Midlands",
    "London": "RP", "Cockney": "RP", "East London": "RP",
    "Irish": "Irish", "Ireland": "Irish", "Dublin": "Irish", "Northern Ireland": "Irish",
}


def check_huggingface_login():
    """Check if user is logged in to Hugging Face, prompt if not."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        print("Already logged in to Hugging Face.")
    except Exception:
        print("=" * 60)
        print("Please login to Hugging Face to download datasets.")
        print("You can create a token at: https://huggingface.co/settings/tokens")
        print("=" * 60)
        subprocess.run([sys.executable, "-m", "huggingface_hub.cli.login"], check=True)


def build_manifest_from_hf_vctk():
    """Build manifest by streaming VCTK from Hugging Face (jspaulsen/vctk)."""
    print("\n" + "=" * 60)
    print("Loading VCTK from Hugging Face (jspaulsen/vctk)...")
    print("=" * 60)

    from datasets import load_dataset

    manifest = []
    speaker_counts = Counter()
    os.makedirs(os.path.join(CURRENT_DIR, "data", "vctk_hf"), exist_ok=True)

    try:
        ds = load_dataset("jspaulsen/vctk", split="train", streaming=True)
        class_samples = Counter()
        MAX_PER_CLASS = 500

        for idx, row in enumerate(ds):
            # Extract fields from HF dataset
            speaker_id = str(row.get("speaker_id", row.get("speaker", f"spk_{idx}")))
            accent_tag = str(row.get("accent", row.get("region", row.get("accents", ""))))
            audio = row.get("audio", row.get("speech", {}))

            if not audio:
                continue

            # Map accent to class
            cls = VCTK_ACCENT_MAP.get(accent_tag)
            if not cls:
                for k, v in VCTK_ACCENT_MAP.items():
                    if k.lower() in accent_tag.lower():
                        cls = v
                        break
            if not cls or cls not in CLASSES or cls == "Irish":
                continue
            if class_samples[cls] >= MAX_PER_CLASS:
                continue

            # Save audio
            spk = speaker_id
            p = os.path.join(CURRENT_DIR, "data", "vctk_hf", f"{spk}_{idx:05d}.wav")
            if not os.path.exists(p):
                sf.write(p, np.array(audio["array"], dtype=np.float32), audio["sampling_rate"])

            manifest.append({
                "path": p,
                "label": cls,
                "speaker": spk,
                "class_idx": CLASSES.index(cls),
            })
            speaker_counts[cls] += 1
            class_samples[cls] += 1

            if len(manifest) % 150 == 0:
                print(f"  Downloaded {len(manifest)} samples...")

            if all(class_samples[c] >= MAX_PER_CLASS for c in ["RP", "Scottish", "Welsh", "Northern", "West_Midlands"]):
                break

        print(f"Built manifest with {len(manifest)} clips from jspaulsen/vctk")
        return manifest

    except Exception as e:
        print(f"  jspaulsen/vctk failed: {e}")
        return []


def build_manifest_from_huggingface():
    """Build manifest by streaming from Hugging Face datasets (fallback)."""
    print("\n" + "=" * 60)
    print("Falling back to Hugging Face English Dialects...")
    print("=" * 60)

    from datasets import load_dataset

    manifest = []
    speaker_counts = Counter()
    os.makedirs(os.path.join(CURRENT_DIR, "data", "vctk_hf"), exist_ok=True)

    # Try CSTR-Edinburgh/vctk first
    try:
        print("Connecting to 'CSTR-Edinburgh/vctk'...")
        ds = load_dataset("CSTR-Edinburgh/vctk", "mic1", split="train", streaming=True)
        class_samples = Counter()
        MAX_PER_CLASS = 500

        for idx, row in enumerate(ds):
            accent_tag = str(row.get("accent", row.get("region", "")))
            cls = VCTK_ACCENT_MAP.get(accent_tag)
            if not cls:
                for k, v in VCTK_ACCENT_MAP.items():
                    if k.lower() in accent_tag.lower():
                        cls = v
                        break
            if not cls or cls not in CLASSES or cls == "Irish":
                continue
            if class_samples[cls] >= MAX_PER_CLASS:
                continue

            spk = str(row.get("speaker_id", f"spk_{cls}"))
            audio = row["audio"]
            p = os.path.join(CURRENT_DIR, "data", "vctk_hf", f"{spk}_{idx:05d}.wav")
            if not os.path.exists(p):
                sf.write(p, np.array(audio["array"], dtype=np.float32), audio["sampling_rate"])

            manifest.append({
                "path": p,
                "label": cls,
                "speaker": spk,
                "class_idx": CLASSES.index(cls),
            })
            speaker_counts[cls] += 1
            class_samples[cls] += 1

            if len(manifest) % 150 == 0:
                print(f"  Downloaded {len(manifest)} samples...")

            if all(class_samples[c] >= MAX_PER_CLASS for c in ["RP", "Scottish", "Welsh", "Northern", "West_Midlands"]):
                break

        if len(manifest) > 100:
            print(f"Built manifest with {len(manifest)} clips from CSTR-Edinburgh/vctk")
            return manifest
    except Exception as e:
        print(f"  CSTR-Edinburgh/vctk failed: {e}")

    # Fallback to ylacombe/english_dialects
    print("\nLoading from 'ylacombe/english_dialects'...")
    DIALECT_MAP = {
        "southern_male": "RP", "southern_female": "RP",
        "scottish_male": "Scottish", "scottish_female": "Scottish",
        "welsh_male": "Welsh", "welsh_female": "Welsh",
        "northern_male": "Northern", "northern_female": "Northern",
        "midlands_male": "West_Midlands", "midlands_female": "West_Midlands",
    }

    for subset, cls in DIALECT_MAP.items():
        try:
            ds = load_dataset("ylacombe/english_dialects", subset, split="train", streaming=True)
            cnt = 0
            for idx, row in enumerate(ds):
                spk = str(row.get("speaker_id", f"spk_{subset}"))
                audio = row["audio"]
                p = os.path.join(CURRENT_DIR, "data", "vctk_hf", f"{subset}_{idx:04d}.wav")
                if not os.path.exists(p):
                    sf.write(p, np.array(audio["array"], dtype=np.float32), audio["sampling_rate"])

                manifest.append({
                    "path": p,
                    "label": cls,
                    "speaker": spk,
                    "class_idx": CLASSES.index(cls),
                })
                speaker_counts[cls] += 1
                cnt += 1
                if cnt >= 250:
                    break
            print(f"  {subset} -> {cls}: {cnt} samples")
        except Exception as se:
            print(f"  Could not load {subset}: {se}")

    return manifest


def add_irish_supplement(manifest):
    """Add Irish English samples from Hugging Face if needed."""
    irish_count = sum(1 for m in manifest if m["label"] == "Irish")
    if irish_count >= 100:
        return manifest

    print(f"\nAdding Irish supplement (current: {irish_count})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ylacombe/english_dialects", "irish_male", split="train", streaming=True)
        cnt = 0
        for idx, row in enumerate(ds):
            spk = str(row.get("speaker_id", f"spk_irish_{idx}"))
            audio = row["audio"]
            p = os.path.join(CURRENT_DIR, "data", "vctk_hf", f"irish_{idx:04d}.wav")
            if not os.path.exists(p):
                sf.write(p, np.array(audio["array"], dtype=np.float32), audio["sampling_rate"])

            manifest.append({
                "path": p,
                "label": "Irish",
                "speaker": spk,
                "class_idx": CLASSES.index("Irish"),
            })
            cnt += 1
            if cnt >= 250:
                break
        print(f"  Added {cnt} Irish samples")
    except Exception as e:
        print(f"  Irish supplement failed: {e}")

    return manifest


def add_common_voice_supplement(manifest):
    """Add Mozilla Common Voice samples for underrepresented accents (Irish, Northern)."""
    irish_count = sum(1 for m in manifest if m["label"] == "Irish")
    northern_count = sum(1 for m in manifest if m["label"] == "Northern")

    if irish_count >= 200 and northern_count >= 200:
        return manifest

    print(f"\nAdding Mozilla Common Voice supplement...")
    print(f"  Current Irish: {irish_count}, Northern: {northern_count}")

    try:
        from datasets import load_dataset

        IRISH_TAGS = {"Ireland", "Irish", "Dublin", "Northern Ireland"}
        NORTHERN_TAGS = {"England", "United Kingdom"}

        cv_extras = []
        for cv_version in ["mozilla-foundation/common_voice_17_0", "mozilla-foundation/common_voice_13_0"]:
            try:
                print(f"  Trying {cv_version}...")
                cv = load_dataset(cv_version, "en", split="train", streaming=True)
                for s in cv.take(50000):
                    accent = s.get("accent", "")
                    if accent in IRISH_TAGS:
                        cv_extras.append({
                            "audio": s["audio"],
                            "speaker": s.get("client_id", "cv_unknown"),
                            "label": "Irish",
                            "class_idx": CLASSES.index("Irish"),
                        })
                    elif accent in NORTHERN_TAGS and len([e for e in cv_extras if e["label"] == "Northern"]) < 200:
                        cv_extras.append({
                            "audio": s["audio"],
                            "speaker": s.get("client_id", "cv_unknown"),
                            "label": "Northern",
                            "class_idx": CLASSES.index("Northern"),
                        })

                    irish_cv = len([e for e in cv_extras if e["label"] == "Irish"])
                    northern_cv = len([e for e in cv_extras if e["label"] == "Northern"])
                    if irish_cv >= 200 and northern_cv >= 200:
                        break

                if cv_extras:
                    break
            except Exception as ve:
                print(f"  {cv_version} failed: {ve}")
                continue

        os.makedirs(os.path.join(CURRENT_DIR, "data", "cv_extras"), exist_ok=True)
        for i, item in enumerate(cv_extras):
            p = os.path.join(CURRENT_DIR, "data", "cv_extras", f"cv_{item['label'].lower()}_{i:04d}.wav")
            sf.write(p, np.array(item["audio"]["array"], dtype=np.float32), item["audio"]["sampling_rate"])
            manifest.append({
                "path": p,
                "label": item["label"],
                "speaker": str(item["speaker"]),
                "class_idx": item["class_idx"],
            })

        irish_added = len([e for e in cv_extras if e["label"] == "Irish"])
        northern_added = len([e for e in cv_extras if e["label"] == "Northern"])
        print(f"  Added {irish_added} Irish + {northern_added} Northern Common Voice samples")

    except Exception as e:
        print(f"  Common Voice supplement failed: {e}")

    return manifest


def speaker_disjoint_splits(manifest, train_ratio=0.75, val_ratio=0.15, seed=42):
    """Split manifest into train/val/test with no speaker overlap."""
    random.seed(seed)
    class_to_speakers = {}
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
    val = [e for e in manifest if e["speaker"] in val_speakers]
    test = [e for e in manifest if e["speaker"] in test_speakers]
    return train, val, test


def save_manifest(manifest, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    print("=" * 60)
    print("AccentSense Data Download & Manifest Builder")
    print("6 UK Regional Accent Classes")
    print("=" * 60)

    # Step 0: Check Hugging Face login
    check_huggingface_login()

    # Step 1: Try jspaulsen/vctk on Hugging Face
    manifest = build_manifest_from_hf_vctk()

    # Step 2: Fallback to other HF datasets if needed
    if len(manifest) < 100:
        print(f"\njspaulsen/vctk yielded only {len(manifest)} samples. Using fallback...")
        manifest = build_manifest_from_huggingface()

    # Step 3: Add Irish supplement
    manifest = add_irish_supplement(manifest)

    # Step 4: Add Common Voice supplement
    manifest = add_common_voice_supplement(manifest)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("Manifest Summary")
    print("=" * 60)
    speaker_counts = Counter(m["label"] for m in manifest)
    print(f"Total samples: {len(manifest)}")
    print(f"Unique speakers: {len(set(m['speaker'] for m in manifest))}")
    print("\nClass distribution:")
    for cls in CLASSES:
        print(f"  {cls:<15}: {speaker_counts.get(cls, 0):5d}")

    # Step 6: Create speaker-disjoint splits
    print("\nCreating speaker-disjoint splits...")
    train, val, test = speaker_disjoint_splits(manifest)
    print(f"  Train: {len(train)} samples ({len(set(m['speaker'] for m in train))} speakers)")
    print(f"  Val:   {len(val)} samples ({len(set(m['speaker'] for m in val))} speakers)")
    print(f"  Test:  {len(test)} samples ({len(set(m['speaker'] for m in test))} speakers)")

    # Step 7: Save manifests
    save_manifest(train, os.path.join(CURRENT_DIR, "data", "manifests", "train.json"))
    save_manifest(val, os.path.join(CURRENT_DIR, "data", "manifests", "val.json"))
    save_manifest(test, os.path.join(CURRENT_DIR, "data", "manifests", "test.json"))

    print("\n" + "=" * 60)
    print("SUCCESS! Manifests saved to Backend/data/manifests/")
    print("=" * 60)
    print("\nNext step: python train_wavlm.py --epochs 20")


if __name__ == "__main__":
    main()
