from __future__ import annotations

import json
import random

from pathlib import Path

import numpy as np
import soundfile as sf

import torch
import torchaudio

from torch.nn.utils.rnn import (
    pad_sequence,
)

from torch.utils.data import Dataset

from src.config import (
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)


def load_manifest(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        data = json.load(
            handle
        )

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            f"Manifest must be a list: "
            f"{path}"
        )

    return data


def make_mono(
    waveform
):

    if waveform.ndim == 1:
        return waveform

    return waveform.mean(
        dim=0
    )


def resample(
    waveform,
    original_sr,
    target_sr,
):

    if (
        original_sr
        == target_sr
    ):

        return waveform

    return (
        torchaudio.functional.resample(
            waveform,
            original_sr,
            target_sr,
        )
    )


# =========================================================
# Moderate augmentation
# =========================================================

def random_gain(
    waveform
):

    gain_db = random.uniform(
        -6.0,
        4.0,
    )

    gain = (
        10.0
        ** (
            gain_db
            / 20.0
        )
    )

    return (
        waveform
        * gain
    )


def add_noise(
    waveform
):

    snr_db = random.uniform(
        12.0,
        30.0,
    )

    signal_rms = (
        waveform
        .pow(2)
        .mean()
        .sqrt()
        .clamp_min(
            1e-6
        )
    )

    noise = torch.randn_like(
        waveform
    )

    noise_rms = (
        noise
        .pow(2)
        .mean()
        .sqrt()
        .clamp_min(
            1e-6
        )
    )

    target_noise = (
        signal_rms
        / (
            10.0
            ** (
                snr_db
                / 20.0
            )
        )
    )

    return (
        waveform
        + noise
        * (
            target_noise
            / noise_rms
        )
    )


def bandlimit(
    waveform,
    sr,
):

    if random.random() < 0.5:

        cutoff = random.uniform(
            4200.0,
            7200.0,
        )

        waveform = (
            torchaudio.functional
            .lowpass_biquad(
                waveform,
                sr,
                cutoff,
            )
        )

    if random.random() < 0.25:

        cutoff = random.uniform(
            60.0,
            180.0,
        )

        waveform = (
            torchaudio.functional
            .highpass_biquad(
                waveform,
                sr,
                cutoff,
            )
        )

    return waveform


def mild_reverb(
    waveform
):

    if waveform.numel() < 32:
        return waveform

    taps = random.randint(
        120,
        600,
    )

    time = torch.arange(
        taps,
        dtype=waveform.dtype,
        device=waveform.device,
    )

    decay = torch.exp(
        -time
        / random.uniform(
            80.0,
            220.0,
        )
    )

    impulse = (
        torch.randn(
            taps,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        * decay
    )

    impulse[0] += 8.0

    impulse = (
        impulse
        / impulse
        .abs()
        .sum()
        .clamp_min(
            1e-6
        )
    )

    signal = waveform.view(
        1,
        1,
        -1,
    )

    kernel = (
        impulse
        .flip(0)
        .view(
            1,
            1,
            -1,
        )
    )

    output = (
        torch.nn.functional.conv1d(
            signal,
            kernel,
            padding=taps - 1,
        )
    )

    return (
        output
        .view(-1)
        [:waveform.numel()]
    )


def speed_perturb(
    waveform,
    sr,
):

    factor = random.uniform(
        0.96,
        1.04,
    )

    temporary_sr = max(
        8000,
        int(
            sr
            * factor
        ),
    )

    waveform = (
        torchaudio.functional.resample(
            waveform,
            sr,
            temporary_sr,
        )
    )

    waveform = (
        torchaudio.functional.resample(
            waveform,
            temporary_sr,
            sr,
        )
    )

    return waveform


class UKAccentDataset(
    Dataset
):

    def __init__(
        self,
        manifest,
        augment=False,
        sample_rate=SAMPLE_RATE,
        segment_seconds=
            SEGMENT_SECONDS,
    ):

        self.manifest = manifest

        self.augment = augment

        self.sample_rate = int(
            sample_rate
        )

        self.segment_samples = int(
            segment_seconds
            * sample_rate
        )

    def __len__(self):

        return len(
            self.manifest
        )

    def crop(
        self,
        waveform,
    ):

        length = (
            waveform.numel()
        )

        if (
            length
            <= self.segment_samples
        ):

            return waveform

        if self.augment:

            start = random.randint(
                0,
                length
                - self.segment_samples,
            )

        else:

            start = (
                length
                - self.segment_samples
            ) // 2

        return waveform[
            start:
            start + self.segment_samples
        ]

    def __getitem__(
        self,
        index,
    ):

        entry = (
            self.manifest[index]
        )

        path = Path(
            entry["path"]
        )

        waveform_np, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )

        waveform = torch.from_numpy(
            np.asarray(
                waveform_np
            ).T
        )

        waveform = make_mono(
            waveform
        )

        waveform = resample(
            waveform,
            int(sr),
            self.sample_rate,
        )

        if waveform.numel() == 0:

            raise RuntimeError(
                f"Empty audio: {path}"
            )

        # Remove DC only.
        #
        # Do not z-normalize because
        # Microsoft's official WavLM Base+
        # feature extractor uses
        # do_normalize=False.

        waveform = (
            waveform
            - waveform.mean()
        )

        if self.augment:

            if random.random() < 0.50:

                waveform = random_gain(
                    waveform
                )

            if random.random() < 0.35:

                waveform = add_noise(
                    waveform
                )

            if random.random() < 0.25:

                waveform = bandlimit(
                    waveform,
                    self.sample_rate,
                )

            if random.random() < 0.20:

                waveform = mild_reverb(
                    waveform
                )

            if random.random() < 0.20:

                waveform = speed_perturb(
                    waveform,
                    self.sample_rate,
                )

        waveform = self.crop(
            waveform
        )

        peak = (
            waveform
            .abs()
            .max()
            .clamp_min(
                1.0
            )
        )

        waveform = (
            waveform
            / peak
        ).clamp(
            -1.0,
            1.0,
        )

        label = int(
            entry[
                "class_idx"
            ]
        )

        return (
            waveform.contiguous(),
            label,
        )


def collate_pad(
    batch
):

    waves, labels = zip(
        *batch
    )

    lengths = torch.tensor(
        [
            waveform.numel()
            for waveform in waves
        ],
        dtype=torch.long,
    )

    padded = pad_sequence(
        waves,
        batch_first=True,
        padding_value=0.0,
    )

    mask = (
        torch.arange(
            padded.size(1)
        ).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    return (
        padded,
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
        mask.long(),
    )