"""
Student model: DistilKoBERT encoder + binary classification head.

DistilKoBERT (monologg/distilkobert) is a 3-layer Korean DistilBERT distilled
from SKTBrain KoBERT.  We add a two-class linear head on top of the [CLS]
pooled representation.

The model is intentionally tiny (~32 M params total with the head) so it can
run efficiently on edge devices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig


STUDENT_HF_ID = "monologg/distilkobert"


class DistilKoBERTClassifier(nn.Module):
    """DistilKoBERT encoder with a dropout + linear classification head.

    Args:
        num_labels: Number of output classes (2 for binary phishing detection).
        dropout: Dropout probability applied before the classifier head.
        freeze_encoder: If True, freeze all encoder weights (train head only).
    """

    def __init__(
        self,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(STUDENT_HF_ID, trust_remote_code=True)
        hidden_size: int = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logits of shape (B, num_labels)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT returns (last_hidden_state, hidden_states?).
        # last_hidden_state[:, 0, :] is the [CLS] token representation.
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        map_location: str | torch.device = "cpu",
    ) -> "DistilKoBERTClassifier":
        model = cls(num_labels=num_labels, dropout=dropout)
        state = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
