from __future__ import annotations

import torch
import torch.nn as nn

from transformers import (
    WavLMModel,
)


class AttentiveStatisticsPooling(
    nn.Module
):

    def __init__(
        self,
        hidden_size,
        attention_size=128,
    ):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                hidden_size,
                attention_size,
            ),

            nn.Tanh(),

            nn.Linear(
                attention_size,
                1,
            ),
        )

    def forward(
        self,
        hidden,
        mask=None,
    ):

        scores = (
            self.attention(
                hidden
            )
            .squeeze(-1)
        )

        if mask is not None:

            scores = (
                scores.masked_fill(
                    ~mask.bool(),
                    -1e4,
                )
            )

        weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
            .unsqueeze(-1)
        )

        mean = torch.sum(
            weights
            * hidden,
            dim=1,
        )

        second = torch.sum(
            weights
            * (
                hidden ** 2
            ),
            dim=1,
        )

        std = (
            second
            - mean ** 2
        ).clamp_min(
            1e-5
        ).sqrt()

        return torch.cat(
            [
                mean,
                std,
            ],
            dim=-1,
        )


class WavLMAccentClassifier(
    nn.Module
):

    def __init__(
        self,
        pretrained_model_name=
            "microsoft/wavlm-base-plus",
        num_classes=6,
        freeze_encoder=True,
        unfreeze_top_k_layers=0,
        dropout=0.25,
    ):

        super().__init__()

        self.backbone = (
            WavLMModel.from_pretrained(
                pretrained_model_name
            )
        )

        hidden_size = (
            self.backbone
            .config
            .hidden_size
        )

        self.asp = (
            AttentiveStatisticsPooling(
                hidden_size
            )
        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                hidden_size * 2
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                num_classes,
            ),
        )

        if freeze_encoder:

            self.freeze_backbone()

        else:

            for parameter in (
                self.backbone
                .parameters()
            ):

                parameter.requires_grad = True

        if (
            unfreeze_top_k_layers
            > 0
        ):

            self.unfreeze_top_k(
                unfreeze_top_k_layers
            )

    def freeze_backbone(
        self
    ):

        for parameter in (
            self.backbone
            .parameters()
        ):

            parameter.requires_grad = False

    def unfreeze_top_k(
        self,
        count,
    ):

        self.freeze_backbone()

        layers = (
            self.backbone
            .encoder
            .layers
        )

        count = max(
            0,
            min(
                int(count),
                len(layers),
            ),
        )

        if count:

            for layer in (
                layers[-count:]
            ):

                for parameter in (
                    layer.parameters()
                ):

                    parameter.requires_grad = True

        # Intentionally keep WavLM's
        # low-level convolutional feature
        # extractor frozen.

    def trainable_parameter_counts(
        self
    ):

        total = sum(
            parameter.numel()

            for parameter
            in self.parameters()
        )

        trainable = sum(
            parameter.numel()

            for parameter
            in self.parameters()

            if parameter.requires_grad
        )

        return (
            trainable,
            total,
        )

    def forward(
        self,
        waveform,
        attention_mask=None,
    ):

        outputs = self.backbone(
            input_values=waveform,

            attention_mask=
                attention_mask,

            return_dict=True,
        )

        hidden = (
            outputs
            .last_hidden_state
        )

        feature_mask = None

        if attention_mask is not None:

            feature_mask = (
                self.backbone
                ._get_feature_vector_attention_mask(
                    hidden.shape[1],
                    attention_mask,
                )
            )

        pooled = self.asp(
            hidden,
            feature_mask,
        )

        logits = (
            self.classifier(
                pooled
            )
        )

        return {
            "logits": logits,
            "embedding": pooled,
        }