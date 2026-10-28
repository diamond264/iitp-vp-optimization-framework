"""
Dataset utilities for distillation training.

Produces three outputs per sample:
  - tokenized input (input_ids, attention_mask)
  - hard label (int)
  - teacher soft probability p_phishing ∈ [0, 1] (float)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PhishingDistilDataset(Dataset):
    """Voice phishing dataset for knowledge-distillation training.

    Args:
        df: DataFrame with at least ``text_col``, ``label_col``, and
            optionally a ``teacher_p`` column (soft label from teacher).
        tokenizer: Tokenizer for the student model.
        text_col: Column name for the raw conversation text.
        label_col: Column name for the binary label (0/1).
        max_length: Maximum token length for the student input.
        teacher_p_col: Column name for teacher soft probability.
            If the column is absent, ``teacher_p`` is set to NaN and the
            training loop has no KD contribution for that sample. Use alpha=0
            for the hard-label-only condition (see train.total_loss).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str = "text",
        label_col: str = "label",
        max_length: int = 512,
        teacher_p_col: str = "teacher_p",
    ) -> None:
        self.texts = df[text_col].fillna("").astype(str).tolist()
        self.labels = df[label_col].fillna(0).astype(int).tolist()
        if teacher_p_col in df.columns:
            self.teacher_ps: list[Optional[float]] = df[teacher_p_col].tolist()
        else:
            self.teacher_ps = [None] * len(self.texts)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_token_type_ids=False,
        )
        item: dict = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        tp = self.teacher_ps[idx]
        item["teacher_p"] = torch.tensor(
            float(tp) if tp is not None else float("nan"), dtype=torch.float
        )
        return item


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length sequences in a batch."""
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        padded_ids[i, : ids.size(0)] = ids
        padded_masks[i, : mask.size(0)] = mask

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_masks,
        "label": torch.stack([b["label"] for b in batch]),
        "teacher_p": torch.stack([b["teacher_p"] for b in batch]),
    }


def load_split_df(
    csv_path: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and split stratified into train / val."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
