"""
AccentSense multi-corpus dataset builder.

Automatic sources:
    ylacombe/english_dialects
    kth-tmh/vctk

Optional local sources:
    IViE
    Mozilla Common Voice British English

Examples:

    python download_data.py --smoke-test

    python download_data.py

    python download_data.py --ivie-root "D:/datasets/IViE"

    python download_data.py \
        --ivie-root "D:/datasets/IViE" \
        --common-voice-root "D:/datasets/common_voice_british"

Outputs:

    data/manifests/all.json
    data/manifests/train.json
    data/manifests/val.json
    data/manifests/test.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from src.config import (
    CLASSES,
    CLASS_TO_IDX,
)


DATA_DIR = CURRENT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"


SLR83_DATASET = "ylacombe/english_dialects"
VCTK_DATASET = "kth-tmh/vctk"


SLR83_CONFIGS = {
    "southern_female": "Southern",
    "southern_male": "Southern",

    "scottish_female": "Scottish",
    "scottish_male": "Scottish",

    "welsh_female": "Welsh",
    "welsh_male": "Welsh",

    "northern_female": "Northern",
    "northern_male": "Northern",

    "midlands_female": "Midlands",
    "midlands_male": "Midlands",

    "irish_male": "Irish",
}


# ---------------------------------------------------------
# Conservative VCTK regional mapping
#
# Unknown English regions are discarded rather than guessed.
# ---------------------------------------------------------

VCTK_SOUTHERN = {
    "southern england",
    "se england",
    "south east england",
    "london",
    "surrey",
    "essex",
    "suffolk",
    "kent",
    "hampshire",
    "oxfordshire",
    "berkshire",
    "east anglia",
}

VCTK_NORTHERN = {
    "manchester",
    "yorkshire",
    "newcastle",
    "ne england",
    "northern england",
    "lancashire",
    "cumbria",
    "liverpool",
    "merseyside",
    "cheshire",
    "stockton-on-tees",
    "york",
}

VCTK_MIDLANDS = {
    "birmingham",
    "nottingham",
    "leicester",
    "midlands",
    "west midlands",
    "staffordshire",
    "derbyshire",
    "warwickshire",
}


# ---------------------------------------------------------
# IViE city mapping
# ---------------------------------------------------------

IVIE_CITY_TO_CLASS = {
    "belfast": "Irish",
    "dublin": "Irish",

    "cardiff": "Welsh",

    "cambridge": "Southern",

    "leeds": "Northern",
    "newcastle": "Northern",
    "liverpool": "Northern",
}


# ---------------------------------------------------------
# Common Voice labels that are safe to use.
#
# "England" is deliberately NOT mapped because it does not
# tell us Northern vs Southern vs Midlands.
# ---------------------------------------------------------

CV_ACCENT_TO_CLASS = {
    "scotland": "Scottish",
    "scottish": "Scottish",

    "wales": "Welsh",
    "welsh": "Welsh",

    "ireland": "Irish",
    "irish": "Irish",
    "northern ireland": "Irish",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["slr83", "vctk"],
        choices=["slr83", "vctk"],
    )

    parser.add_argument(
        "--ivie-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--common-voice-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_per_speaker < 1:
        parser.error("--max-per-speaker must be positive")

    if args.max_per_class < 1:
        parser.error("--max-per-class must be positive")

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")

    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1")

    if args.train_ratio + args.val_ratio >= 1:
        parser.error("train + val ratio must be below 1")

    if args.smoke_test:
        args.max_per_speaker = min(
            args.max_per_speaker,
            2,
        )

        args.max_per_class = min(
            args.max_per_class,
            48,
        )

    return args


def norm(value):

    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip().lower(),
    )


def safe_token(value):

    result = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    )

    return result.strip("._-") or "unknown"


# =========================================================
# Hugging Face audio without TorchCodec
# =========================================================

def audio_from_decode_false(value):

    if not isinstance(value, dict):

        raise TypeError(
            "Expected Audio(decode=False) dictionary"
        )

    raw = value.get("bytes")
    path = value.get("path")

    if raw is not None:

        if isinstance(raw, memoryview):
            raw = raw.tobytes()

        with io.BytesIO(bytes(raw)) as handle:

            waveform, sr = sf.read(
                handle,
                dtype="float32",
                always_2d=False,
            )

        return waveform, int(sr)

    if path and os.path.exists(path):

        waveform, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        return waveform, int(sr)

    if path:

        import fsspec

        with fsspec.open(
            path,
            "rb",
        ) as handle:

            waveform, sr = sf.read(
                handle,
                dtype="float32",
                always_2d=False,
            )

        return waveform, int(sr)

    raise ValueError(
        "No audio bytes/path were available"
    )


def write_hf_audio(audio_value, destination):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():

        try:

            info = sf.info(
                str(destination)
            )

            if info.frames > 0:
                return

        except Exception:
            pass

    waveform, sr = audio_from_decode_false(
        audio_value
    )

    temporary = destination.with_suffix(
        ".part.wav"
    )

    sf.write(
        str(temporary),
        waveform,
        sr,
        subtype="PCM_16",
    )

    info = sf.info(
        str(temporary)
    )

    if info.frames <= 0:

        raise RuntimeError(
            "Written audio failed validation"
        )

    os.replace(
        temporary,
        destination,
    )


# =========================================================
# Local audio conversion
# =========================================================

def convert_local_audio(source, destination):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():

        try:

            if sf.info(
                str(destination)
            ).frames > 0:

                return

        except Exception:
            pass

    try:

        waveform, sr = sf.read(
            str(source),
            dtype="float32",
            always_2d=False,
        )

        temporary = destination.with_suffix(
            ".part.wav"
        )

        sf.write(
            str(temporary),
            waveform,
            sr,
            subtype="PCM_16",
        )

        os.replace(
            temporary,
            destination,
        )

        return

    except Exception:
        pass

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:

        raise RuntimeError(
            f"Could not decode {source}. "
            "FFmpeg was not found."
        )

    temporary = destination.with_suffix(
        ".tmp.wav"
    )

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",

            "-i",
            str(source),

            "-ac",
            "1",

            str(temporary),
        ],
        check=True,
    )

    if sf.info(
        str(temporary)
    ).frames <= 0:

        raise RuntimeError(
            f"Invalid converted audio: {source}"
        )

    os.replace(
        temporary,
        destination,
    )


# =========================================================
# HF streaming
# =========================================================

def hf_stream(
    dataset_name,
    config=None,
    retries=3,
):

    from datasets import load_dataset

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            dataset = load_dataset(
                dataset_name,
                config,
                split="train",
                streaming=True,
            )

            if not hasattr(
                dataset,
                "decode",
            ):

                raise RuntimeError(
                    "datasets package does not "
                    "support decode(False)"
                )

            return dataset.decode(
                False
            )

        except Exception as exc:

            last_error = exc

            if attempt == retries:
                break

            time.sleep(
                min(
                    2 ** (attempt - 1),
                    8,
                )
            )

    raise RuntimeError(
        f"Unable to load "
        f"{dataset_name}/{config}: "
        f"{last_error}"
    ) from last_error


# =========================================================
# Manifest helper
# =========================================================

def add_entry(
    manifest,
    class_counts,
    speaker_counts,
    *,
    source,
    label,
    speaker,
    audio_path,
    max_per_class,
    max_per_speaker,
    **metadata,
):

    if label not in CLASS_TO_IDX:
        return False

    if (
        class_counts[label]
        >= max_per_class
    ):
        return False

    if (
        speaker_counts[speaker]
        >= max_per_speaker
    ):
        return False

    manifest.append(
        {
            "path": str(
                audio_path.resolve()
            ),

            "label": label,

            "class_idx":
                CLASS_TO_IDX[label],

            "speaker": speaker,

            "source": source,

            **metadata,
        }
    )

    class_counts[label] += 1
    speaker_counts[speaker] += 1

    return True


# =========================================================
# SLR83
# =========================================================

def collect_slr83(
    manifest,
    class_counts,
    speaker_counts,
    args,
):

    print(
        "\n[SLR83 / english_dialects]"
    )

    for config, label in (
        SLR83_CONFIGS.items()
    ):

        dataset = hf_stream(
            SLR83_DATASET,
            config,
            args.retries,
        )

        before = class_counts[label]

        for index, row in enumerate(
            dataset
        ):

            if (
                class_counts[label]
                >= args.max_per_class
            ):
                break

            raw_speaker = row.get(
                "speaker_id"
            )

            audio = row.get(
                "audio"
            )

            if (
                raw_speaker is None
                or not audio
            ):
                continue

            speaker = (
                f"slr83:{config}:{raw_speaker}"
            )

            if (
                speaker_counts[speaker]
                >= args.max_per_speaker
            ):
                continue

            line_id = row.get(
                "line_id",
                index,
            )

            destination = (
                AUDIO_DIR
                / "slr83"
                / label
                / safe_token(raw_speaker)
                / f"{safe_token(line_id)}.wav"
            )

            try:

                write_hf_audio(
                    audio,
                    destination,
                )

            except Exception as exc:

                print(
                    f"Skipping "
                    f"{config}/{index}: "
                    f"{exc}"
                )

                continue

            add_entry(
                manifest,
                class_counts,
                speaker_counts,

                source="slr83",

                label=label,

                speaker=speaker,

                audio_path=destination,

                max_per_class=
                    args.max_per_class,

                max_per_speaker=
                    args.max_per_speaker,

                original_group=config,
            )

        added = (
            class_counts[label]
            - before
        )

        print(
            f"{config:<20} +{added}"
        )


# =========================================================
# VCTK mapping
# =========================================================

def map_vctk(
    accent,
    region,
):

    accent = norm(
        accent
    )

    region = norm(
        region
    )

    if accent == "scottish":
        return "Scottish"

    if accent == "welsh":
        return "Welsh"

    if accent in {
        "irish",
        "northern irish",
    }:
        return "Irish"

    if accent != "english":
        return None

    if region in VCTK_SOUTHERN:
        return "Southern"

    if region in VCTK_NORTHERN:
        return "Northern"

    if region in VCTK_MIDLANDS:
        return "Midlands"

    return None


def collect_vctk(
    manifest,
    class_counts,
    speaker_counts,
    args,
):

    print("\n[VCTK]")

    dataset = hf_stream(
        VCTK_DATASET,
        "default",
        args.retries,
    )

    unmapped = Counter()

    for index, row in enumerate(dataset):

        source_file = str(
            row.get(
                "file",
                "",
            )
        )

        # Keep only mic1 to avoid duplicate utterances
        if "_mic2." in source_file.lower():
            continue

        raw_speaker = row.get("speaker_id")
        audio = row.get("audio")

        if raw_speaker is None or not audio:
            continue

        label = map_vctk(
            row.get("accent"),
            row.get("region"),
        )

        if not label:
            unmapped[
                (
                    str(row.get("accent")),
                    str(row.get("region")),
                )
            ] += 1
            continue

        if class_counts[label] >= args.max_per_class:
            continue

        speaker = f"vctk:{raw_speaker}"

        if speaker_counts[speaker] >= args.max_per_speaker:
            continue

        utterance = row.get(
            "text_id",
            index,
        )

        destination = (
            AUDIO_DIR
            / "vctk"
            / label
            / safe_token(raw_speaker)
            / f"{safe_token(utterance)}.wav"
        )

        try:
            write_hf_audio(
                audio,
                destination,
            )

        except Exception as exc:
            print(
                f"Skipping VCTK row {index}: {exc}"
            )
            continue

        add_entry(
            manifest,
            class_counts,
            speaker_counts,

            source="vctk",
            label=label,
            speaker=speaker,
            audio_path=destination,

            max_per_class=args.max_per_class,
            max_per_speaker=args.max_per_speaker,

            raw_accent=str(
                row.get(
                    "accent",
                    "",
                )
            ),

            raw_region=str(
                row.get(
                    "region",
                    "",
                )
            ),
        )

    if unmapped:
        print("\nTop unmapped VCTK metadata:")

        for key, count in unmapped.most_common(10):
            print(
                f"  {key}: {count}"
            )


# =========================================================
# IViE
# =========================================================

def find_ivie_city(path):

    joined = "/".join(
        part.lower()
        for part in path.parts
    )

    for city in IVIE_CITY_TO_CLASS:

        if city in joined:
            return city

    return None


def ivie_speaker_from_name(
    stem
):

    # Revised IViE names contain speaker identifiers
    # such as b-rea1a-f1, b-rea3-m2 etc.

    match = re.search(
        r"(?:^|[-_])([fm])(\d+)(?:$|[-_])",
        stem.lower(),
    )

    if not match:
        return None

    return (
        f"{match.group(1)}"
        f"{match.group(2)}"
    )


def collect_ivie(
    root,
    manifest,
    class_counts,
    speaker_counts,
    args,
):

    print(
        f"\n[IViE: {root}]"
    )

    if not root.exists():

        raise FileNotFoundError(
            root
        )

    accepted = 0
    unsafe_names = 0

    for source in root.rglob(
        "*.wav"
    ):

        city = find_ivie_city(
            source
        )

        if not city:
            continue

        label = (
            IVIE_CITY_TO_CLASS[
                city
            ]
        )

        speaker_token = (
            ivie_speaker_from_name(
                source.stem
            )
        )

        if not speaker_token:

            unsafe_names += 1
            continue

        speaker = (
            f"ivie:{city}:"
            f"{speaker_token}"
        )

        if (
            speaker_counts[speaker]
            >= args.max_per_speaker
        ):
            continue

        destination = (
            AUDIO_DIR
            / "ivie"
            / label
            / safe_token(speaker)
            / f"{safe_token(source.stem)}.wav"
        )

        try:

            convert_local_audio(
                source,
                destination,
            )

        except Exception as exc:

            print(
                f"Skipping IViE "
                f"{source}: {exc}"
            )

            continue

        if add_entry(
            manifest,
            class_counts,
            speaker_counts,

            source="ivie",

            label=label,

            speaker=speaker,

            audio_path=destination,

            max_per_class=
                args.max_per_class,

            max_per_speaker=
                args.max_per_speaker,

            raw_region=city,
        ):

            accepted += 1

    print(
        f"Accepted {accepted} IViE clips"
    )

    if unsafe_names:

        print(
            f"Skipped {unsafe_names} IViE "
            "files because a reliable speaker "
            "identifier could not be inferred."
        )


# =========================================================
# Common Voice
# =========================================================

def common_voice_label(
    value
):

    accent = norm(
        value
    )

    if not accent:
        return None

    # Generic England cannot tell us which
    # English regional group the speaker belongs to.

    if (
        "england" in accent
        and not any(
            token in accent
            for token in (
                "scot",
                "welsh",
                "wales",
                "irish",
                "ireland",
            )
        )
    ):
        return None

    for token, label in (
        CV_ACCENT_TO_CLASS.items()
    ):

        if token in accent:
            return label

    return None


def collect_common_voice(
    root,
    manifest,
    class_counts,
    speaker_counts,
    args,
):

    print(
        f"\n[Common Voice: {root}]"
    )

    if not root.exists():

        raise FileNotFoundError(
            root
        )

    candidates = [
        root / "validated.tsv",
        root / "train.tsv",
        root / "validated.csv",
    ]

    metadata = next(
        (
            candidate
            for candidate in candidates
            if candidate.exists()
        ),
        None,
    )

    if metadata is None:

        raise FileNotFoundError(
            "No validated.tsv/train.tsv "
            "was found in Common Voice root."
        )

    delimiter = (
        "\t"
        if metadata.suffix.lower()
        == ".tsv"
        else ","
    )

    accepted = 0

    with metadata.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
        )

        for row in reader:

            raw_speaker = (
                row.get("client_id")
                or row.get("speaker_id")
            )

            relative_path = (
                row.get("path")
                or row.get("filename")
            )

            accent = (
                row.get("accents")
                or row.get("accent")
                or row.get("variant")
            )

            label = common_voice_label(
                accent
            )

            if (
                not raw_speaker
                or not relative_path
                or not label
            ):
                continue

            speaker = (
                f"commonvoice:"
                f"{raw_speaker}"
            )

            if (
                speaker_counts[speaker]
                >= args.max_per_speaker
            ):
                continue

            source = (
                root
                / "clips"
                / relative_path
            )

            if not source.exists():

                source = (
                    root
                    / relative_path
                )

            if not source.exists():
                continue

            destination = (
                AUDIO_DIR
                / "commonvoice"
                / label
                / safe_token(raw_speaker)
                / (
                    f"{safe_token(Path(relative_path).stem)}"
                    ".wav"
                )
            )

            try:

                convert_local_audio(
                    source,
                    destination,
                )

            except Exception as exc:

                print(
                    f"Skipping "
                    f"{source.name}: "
                    f"{exc}"
                )

                continue

            if add_entry(
                manifest,
                class_counts,
                speaker_counts,

                source="commonvoice",

                label=label,

                speaker=speaker,

                audio_path=destination,

                max_per_class=
                    args.max_per_class,

                max_per_speaker=
                    args.max_per_speaker,

                raw_accent=str(
                    accent or ""
                ),
            ):

                accepted += 1

    print(
        f"Accepted {accepted} "
        "Common Voice clips"
    )


# =========================================================
# Validation
# =========================================================

def validate_manifest(
    manifest
):

    if not manifest:

        raise RuntimeError(
            "Manifest is empty"
        )

    paths = set()
    speaker_labels = {}

    speakers_per_class = (
        defaultdict(set)
    )

    for entry in manifest:

        label = entry[
            "label"
        ]

        speaker = entry[
            "speaker"
        ]

        path = entry[
            "path"
        ]

        if label not in CLASS_TO_IDX:

            raise RuntimeError(
                f"Unknown class: {label}"
            )

        if (
            entry["class_idx"]
            != CLASS_TO_IDX[label]
        ):

            raise RuntimeError(
                "class_idx mismatch"
            )

        if not Path(
            path
        ).is_file():

            raise RuntimeError(
                f"Missing audio: {path}"
            )

        if path in paths:

            raise RuntimeError(
                f"Duplicate path: {path}"
            )

        paths.add(
            path
        )

        previous = (
            speaker_labels.setdefault(
                speaker,
                label,
            )
        )

        if previous != label:

            raise RuntimeError(
                f"Speaker {speaker} "
                "has conflicting labels"
            )

        speakers_per_class[
            label
        ].add(
            speaker
        )

    missing = [
        class_name
        for class_name in CLASSES
        if not speakers_per_class[
            class_name
        ]
    ]

    if missing:

        raise RuntimeError(
            f"Missing classes: {missing}"
        )

    too_small = {
        class_name:
            len(
                speakers_per_class[
                    class_name
                ]
            )

        for class_name in CLASSES

        if len(
            speakers_per_class[
                class_name
            ]
        ) < 3
    }

    if too_small:

        raise RuntimeError(
            "Need at least 3 real speakers "
            "per class for train/val/test. "
            f"Insufficient: {too_small}"
        )


# =========================================================
# Speaker-disjoint splitting
# =========================================================

def speaker_disjoint_splits(
    manifest,
    train_ratio,
    val_ratio,
    seed,
):

    speaker_entries = (
        defaultdict(list)
    )

    speaker_label = {}

    for entry in manifest:

        speaker = entry[
            "speaker"
        ]

        label = entry[
            "label"
        ]

        if (
            speaker in speaker_label
            and speaker_label[speaker]
            != label
        ):

            raise RuntimeError(
                f"Conflicting labels "
                f"for {speaker}"
            )

        speaker_label[
            speaker
        ] = label

        speaker_entries[
            speaker
        ].append(
            entry
        )

    by_class = (
        defaultdict(list)
    )

    for speaker, label in (
        speaker_label.items()
    ):

        by_class[
            label
        ].append(
            speaker
        )

    split_speakers = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    test_ratio = (
        1.0
        - train_ratio
        - val_ratio
    )

    for class_index, class_name in enumerate(
        CLASSES
    ):

        speakers = sorted(
            by_class[
                class_name
            ]
        )

        if len(speakers) < 3:

            raise RuntimeError(
                f"{class_name} only "
                f"has {len(speakers)} speakers"
            )

        rng = random.Random(
            seed + class_index
        )

        rng.shuffle(
            speakers
        )

        count = len(
            speakers
        )

        n_val = max(
            1,
            round(
                count
                * val_ratio
            ),
        )

        n_test = max(
            1,
            round(
                count
                * test_ratio
            ),
        )

        while (
            n_val
            + n_test
            >= count
        ):

            if (
                n_val >= n_test
                and n_val > 1
            ):

                n_val -= 1

            elif n_test > 1:

                n_test -= 1

            else:
                break

        n_train = (
            count
            - n_val
            - n_test
        )

        split_speakers[
            "train"
        ].update(
            speakers[
                :n_train
            ]
        )

        split_speakers[
            "val"
        ].update(
            speakers[
                n_train:
                n_train + n_val
            ]
        )

        split_speakers[
            "test"
        ].update(
            speakers[
                n_train + n_val:
            ]
        )

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):

        overlap = (
            split_speakers[left]
            & split_speakers[right]
        )

        if overlap:

            raise RuntimeError(
                f"Speaker leakage "
                f"{left}/{right}: "
                f"{sorted(overlap)[:5]}"
            )

    output = {
        "train": [],
        "val": [],
        "test": [],
    }

    for speaker, entries in (
        speaker_entries.items()
    ):

        destination = next(
            split_name

            for split_name, members
            in split_speakers.items()

            if speaker in members
        )

        output[
            destination
        ].extend(
            entries
        )

    for split_name in output:

        output[
            split_name
        ].sort(
            key=lambda entry: (
                entry["label"],
                entry["speaker"],
                entry["path"],
            )
        )

    return output


def save_json(
    data,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def print_summary(
    manifest,
    splits,
):

    print(
        "\n"
        + "=" * 76
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 76
    )

    for class_name in CLASSES:

        rows = [
            entry

            for entry in manifest

            if entry[
                "label"
            ] == class_name
        ]

        speakers = {
            entry[
                "speaker"
            ]

            for entry in rows
        }

        sources = Counter(
            entry[
                "source"
            ]

            for entry in rows
        )

        print(
            f"{class_name:<10} "
            f"{len(rows):5d} clips | "
            f"{len(speakers):3d} speakers | "
            f"{dict(sources)}"
        )

    print(
        "\nSplits:"
    )

    for name, rows in (
        splits.items()
    ):

        speakers = {
            entry[
                "speaker"
            ]

            for entry in rows
        }

        print(
            f"{name:<5}: "
            f"{len(rows):5d} clips | "
            f"{len(speakers):3d} speakers"
        )


def main():

    args = parse_args()

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    class_counts = Counter()
    speaker_counts = Counter()

    if "slr83" in args.sources:

        collect_slr83(
            manifest,
            class_counts,
            speaker_counts,
            args,
        )

    if "vctk" in args.sources:

        collect_vctk(
            manifest,
            class_counts,
            speaker_counts,
            args,
        )

    if args.ivie_root:

        collect_ivie(
            Path(
                args.ivie_root
            ),
            manifest,
            class_counts,
            speaker_counts,
            args,
        )

    if args.common_voice_root:

        collect_common_voice(
            Path(
                args.common_voice_root
            ),
            manifest,
            class_counts,
            speaker_counts,
            args,
        )

    validate_manifest(
        manifest
    )

    splits = (
        speaker_disjoint_splits(
            manifest,

            train_ratio=
                args.train_ratio,

            val_ratio=
                args.val_ratio,

            seed=args.seed,
        )
    )

    print_summary(
        manifest,
        splits,
    )

    save_json(
        manifest,
        MANIFEST_DIR / "all.json",
    )

    save_json(
        splits["train"],
        MANIFEST_DIR / "train.json",
    )

    save_json(
        splits["val"],
        MANIFEST_DIR / "val.json",
    )

    save_json(
        splits["test"],
        MANIFEST_DIR / "test.json",
    )

    print(
        "\nSUCCESS"
    )

    print(
        f"Manifests: {MANIFEST_DIR}"
    )

    print(
        "Speaker overlap across splits: 0"
    )


if __name__ == "__main__":
    main()