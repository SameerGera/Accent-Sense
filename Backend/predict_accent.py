from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile

from pathlib import Path

import numpy as np
import soundfile as sf

import torch
import torchaudio

from src.config import (
    CLASSES,
    DISPLAY_NAMES,
    SAMPLE_RATE,
)

from src.models.wavlm_classifier import (
    WavLMAccentClassifier,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio"
    )

    parser.add_argument(
        "--checkpoint",
        default=
            "checkpoints/"
            "best_wavlm_accentsense.pt",
    )

    parser.add_argument(
        "--window_seconds",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min_rms",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--uncertain_threshold",
        type=float,
        default=0.45,
    )

    return parser.parse_args()


def read_audio(
    path
):

    try:

        waveform, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )

        waveform = (
            torch.from_numpy(
                np.asarray(
                    waveform
                ).T
            )
            .mean(
                dim=0
            )
        )

        return (
            waveform,
            int(sr),
        )

    except Exception:
        pass

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:

        raise RuntimeError(
            "Could not decode file "
            "and FFmpeg is unavailable."
        )

    with tempfile.TemporaryDirectory() as temp:

        converted = (
            Path(temp)
            / "decoded.wav"
        )

        subprocess.run(
            [
                ffmpeg,

                "-hide_banner",
                "-loglevel",
                "error",
                "-y",

                "-i",
                str(path),

                "-ac",
                "1",

                "-ar",
                str(SAMPLE_RATE),

                str(converted),
            ],
            check=True,
        )

        waveform, sr = sf.read(
            str(converted),
            dtype="float32",
        )

        return (
            torch.tensor(
                waveform,
                dtype=torch.float32,
            ),
            int(sr),
        )


def make_chunks(
    waveform,
    window,
    hop,
    min_rms,
):

    if (
        waveform.numel()
        <= window
    ):

        candidates = [
            waveform
        ]

    else:

        starts = list(
            range(
                0,

                max(
                    1,
                    waveform.numel()
                    - window
                    + 1,
                ),

                hop,
            )
        )

        if (
            starts[-1]
            + window
            < waveform.numel()
        ):

            starts.append(
                waveform.numel()
                - window
            )

        candidates = [
            waveform[
                start:
                start + window
            ]

            for start in starts
        ]

    chunks = []

    for chunk in candidates:

        if not chunk.numel():
            continue

        rms = (
            chunk
            .pow(2)
            .mean()
            .sqrt()
            .item()
        )

        if rms < min_rms:
            continue

        if chunk.numel() < window:

            chunk = (
                torch.nn.functional.pad(
                    chunk,
                    (
                        0,
                        window
                        - chunk.numel(),
                    ),
                )
            )

        chunks.append(
            chunk
        )

    return chunks


def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    classes = tuple(
        checkpoint.get(
            "classes",
            CLASSES,
        )
    )

    model_name = (
        checkpoint.get(
            "model_name",
            "microsoft/wavlm-base-plus",
        )
    )

    model = (
        WavLMAccentClassifier(

            pretrained_model_name=
                model_name,

            num_classes=
                len(classes),

            freeze_encoder=True,

            unfreeze_top_k_layers=0,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    waveform, sr = read_audio(
        Path(
            args.audio
        )
    )

    if sr != SAMPLE_RATE:

        waveform = (
            torchaudio.functional
            .resample(
                waveform,
                sr,
                SAMPLE_RATE,
            )
        )

    waveform = (
        waveform
        - waveform.mean()
    )

    window = int(
        args.window_seconds
        * SAMPLE_RATE
    )

    hop = int(
        args.hop_seconds
        * SAMPLE_RATE
    )

    chunks = make_chunks(
        waveform,
        window,
        hop,
        args.min_rms,
    )

    if not chunks:

        raise RuntimeError(
            "No usable speech-like "
            "windows detected."
        )

    probabilities = []

    with torch.no_grad():

        for chunk in chunks:

            batch = (
                chunk
                .unsqueeze(0)
                .to(
                    device
                )
            )

            mask = torch.ones_like(
                batch,
                dtype=torch.long,
            )

            logits = model(
                batch,
                attention_mask=mask,
            )["logits"]

            probability = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
                .cpu()
            )

            probabilities.append(
                probability
            )

    mean_probability = (
        torch.cat(
            probabilities,
            dim=0,
        )
        .mean(
            dim=0
        )
    )

    order = torch.argsort(
        mean_probability,
        descending=True,
    )

    top_index = int(
        order[0]
    )

    confidence = float(
        mean_probability[
            top_index
        ]
    )

    label = classes[
        top_index
    ]

    result = {

        "accent":
            label,

        "display_name":
            DISPLAY_NAMES.get(
                label,
                label,
            ),

        "confidence":
            confidence,

        "uncertain":
            (
                confidence
                < args.uncertain_threshold
            ),

        "windows_used":
            len(chunks),

        "probabilities": {
            classes[int(index)]:
                float(
                    mean_probability[
                        int(index)
                    ]
                )

            for index in order
        },
    }

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()