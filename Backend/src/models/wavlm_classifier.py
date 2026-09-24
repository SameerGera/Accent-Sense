"""
WavLM Base+ with Attentive Statistics Pooling for UK Regional Accent Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, Dict, Tuple, Any


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) layer.
    Computes both attention-weighted mean and attention-weighted standard deviation
    to capture steady-state vowel formants and rapid phonetic transitions.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [Batch, Time, HiddenDim]
        Returns:
            pooled: [Batch, HiddenDim * 2]
            alpha:  [Batch, Time, 1] (attention weights for explainability)
        """
        scores = self.attention(hidden_states)  # [Batch, Time, 1]

        if attention_mask is not None:
            mask_float = attention_mask.float().unsqueeze(-1)
            scores = scores * mask_float + (1.0 - mask_float) * (-1e9)

        alpha = F.softmax(scores, dim=1)  # [Batch, Time, 1]

        mean = torch.sum(alpha * hidden_states, dim=1)
        residual = hidden_states - mean.unsqueeze(1)
        var = torch.sum(alpha * (residual ** 2), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-6))

        pooled = torch.cat([mean, std], dim=-1)  # [Batch, HiddenDim * 2]
        return pooled, alpha


class WavLMAccentClassifier(nn.Module):
    """
    End-to-end UK regional accent classifier:
    WavLM Base+ -> Attentive Statistics Pooling -> Classification Head.

    Supports frozen encoder with optional top-K layer fine-tuning.
    Default: 6 UK classes (RP, Scottish, Welsh, Northern, West_Midlands, Irish).
    """
    def __init__(
        self,
        pretrained_model_name: str = "microsoft/wavlm-base-plus",
        num_classes: int = 6,
        freeze_encoder: bool = True,
        unfreeze_top_k_layers: int = 2,
        dropout_p: float = 0.3,
        config: Optional[Any] = None,
    ):
        super().__init__()
        self.model_name = pretrained_model_name
        self.freeze_encoder = freeze_encoder
        self.unfreeze_top_k_layers = unfreeze_top_k_layers
        if config is not None:
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        hidden_dim = self.backbone.config.hidden_size

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_top_k_layers > 0 and hasattr(self.backbone, "encoder"):
                total_layers = len(self.backbone.encoder.layers)
                for i in range(total_layers - unfreeze_top_k_layers, total_layers):
                    for param in self.backbone.encoder.layers[i].parameters():
                        param.requires_grad = True

        self.asp = AttentiveStatisticsPooling(hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout_p / 2),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: Raw audio waveforms [Batch, Time] at 16 kHz mono.
        Returns:
            dict: logits, probabilities, attention_weights,
                  optionally frame_features and all_hidden_states.
        """
        is_dml = input_values.device.type == "privateuseone"
        all_hidden_states = []

        if self.freeze_encoder and hasattr(self.backbone, "encoder") and not return_hidden:
            # Memory- & DirectML-optimized path: run frozen CNN + frozen bottom Transformer layers under torch.no_grad()
            total_layers = len(self.backbone.encoder.layers)
            k = max(0, min(self.unfreeze_top_k_layers, total_layers))
            frozen_layers = self.backbone.encoder.layers[: total_layers - k]
            trainable_layers = self.backbone.encoder.layers[total_layers - k :]

            with torch.no_grad():
                extract_features = self.backbone.feature_extractor(input_values)
                extract_features = extract_features.transpose(1, 2)
                hidden_states, _ = self.backbone.feature_projection(extract_features)
                pos_conv = self.backbone.encoder.pos_conv_embed(hidden_states)
                hidden_states = hidden_states + pos_conv
                hidden_states = self.backbone.encoder.layer_norm(hidden_states)
                hidden_states = self.backbone.encoder.dropout(hidden_states)

                pos_bias = None
                for layer in frozen_layers:
                    layer_out = layer(hidden_states, position_bias=pos_bias)
                    hidden_states = layer_out[0]
                    pos_bias = layer_out[1]

            hidden_states = hidden_states.detach()
            if pos_bias is not None:
                pos_bias = pos_bias.detach()

            for layer in trainable_layers:
                layer_out = layer(hidden_states, position_bias=pos_bias)
                hidden_states = layer_out[0]
                pos_bias = layer_out[1]

            frame_features = hidden_states
        else:
            backbone_mask = None if is_dml else attention_mask
            outputs = self.backbone(
                input_values,
                attention_mask=backbone_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            frame_features = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states

        feat_mask = None
        if attention_mask is not None:
            mask_for_calc = attention_mask.cpu() if is_dml else attention_mask
            if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
                feat_mask = self.backbone._get_feature_vector_attention_mask(
                    frame_features.shape[1], mask_for_calc
                ).float().to(input_values.device)
            elif attention_mask.shape[1] == frame_features.shape[1]:
                feat_mask = attention_mask.float()
            else:
                feat_mask = F.interpolate(
                    mask_for_calc.unsqueeze(1).float(),
                    size=frame_features.shape[1],
                    mode="nearest",
                ).squeeze(1).to(input_values.device)

            if is_dml:
                frame_features = frame_features * feat_mask.unsqueeze(-1)

        pooled_features, attention_weights = self.asp(frame_features, attention_mask=feat_mask)
        logits = self.classifier(pooled_features)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "attention_weights": attention_weights,
        }
        if return_hidden:
            result["frame_features"] = frame_features
            result["all_hidden_states"] = all_hidden_states

        return result


# Backward-compatibility alias (old name used in checkpoints)
WavLMForL1Influence = WavLMAccentClassifier
