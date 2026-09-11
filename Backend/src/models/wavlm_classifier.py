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
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

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
    Default: 7 UK classes (RP, Scottish, Welsh, Northern, West_Midlands, Cockney, Irish).
    """
    def __init__(
        self,
        pretrained_model_name: str = "microsoft/wavlm-base-plus",
        num_classes: int = 7,
        freeze_encoder: bool = True,
        unfreeze_top_k_layers: int = 2,
        dropout_p: float = 0.3,
        config: Optional[Any] = None,
    ):
        super().__init__()
        self.model_name = pretrained_model_name
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
        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        frame_features = outputs.last_hidden_state
        pooled_features, attention_weights = self.asp(frame_features)
        logits = self.classifier(pooled_features)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "attention_weights": attention_weights,
        }
        if return_hidden:
            result["frame_features"] = frame_features
            result["all_hidden_states"] = outputs.hidden_states

        return result


# Backward-compatibility alias (old name used in checkpoints)
WavLMForL1Influence = WavLMAccentClassifier
