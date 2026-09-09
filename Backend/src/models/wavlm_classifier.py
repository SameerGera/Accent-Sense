"""
WavLM Base+ with Attentive Statistics Pooling for Native Language Influence Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2Model, AutoModel
from typing import Optional, Dict, Tuple, Any


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) layer.
    Computes both attention-weighted mean and attention-weighted standard deviation
    to capture both steady-state vowel formants and rapid phonetic transitions.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [Batch, Time, HiddenDim]
        Returns:
            pooled: [Batch, HiddenDim * 2]
            alpha: [Batch, Time, 1] (attention weights for explainability)
        """
        # Calculate raw attention scores
        scores = self.attention(hidden_states)  # [Batch, Time, 1]
        
        if attention_mask is not None:
            # Mask out padding frames
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            
        alpha = F.softmax(scores, dim=1)  # [Batch, Time, 1]

        # Weighted Mean
        mean = torch.sum(alpha * hidden_states, dim=1)  # [Batch, HiddenDim]

        # Weighted Variance and Std Dev
        residual = hidden_states - mean.unsqueeze(1)
        var = torch.sum(alpha * (residual ** 2), dim=1)  # [Batch, HiddenDim]
        std = torch.sqrt(torch.clamp(var, min=1e-6))    # [Batch, HiddenDim]

        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=-1)  # [Batch, HiddenDim * 2]
        return pooled, alpha


class WavLMForL1Influence(nn.Module):
    """
    End-to-end architecture:
    WavLM Backbone -> ASP Pooling -> Classification Head.
    Supports frozen encoder, partial fine-tuning, and extraction of intermediate layer representations.
    """
    def __init__(
        self,
        pretrained_model_name: str = "microsoft/wavlm-base-plus",
        num_classes: int = 4,
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

        # Freeze strategy
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # If specified, unfreeze top K transformer layers
            if unfreeze_top_k_layers > 0 and hasattr(self.backbone, "encoder"):
                total_layers = len(self.backbone.encoder.layers)
                for i in range(total_layers - unfreeze_top_k_layers, total_layers):
                    for param in self.backbone.encoder.layers[i].parameters():
                        param.requires_grad = True

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(hidden_dim=hidden_dim)

        # Classification Head
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
            input_values: Raw audio waveforms [Batch, Time] (16kHz)
        """
        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract last hidden state or intermediate representations
        frame_features = outputs.last_hidden_state  # [Batch, FrameTime, HiddenDim]

        # Apply Attentive Statistics Pooling
        pooled_features, attention_weights = self.asp(frame_features)

        # Compute logits
        logits = self.classifier(pooled_features)

        result = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "attention_weights": attention_weights,
        }

        if return_hidden:
            result["frame_features"] = frame_features
            result["all_hidden_states"] = outputs.hidden_states

        return result
