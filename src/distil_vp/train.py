"""
Distillation training script.

Loss:
    L = alpha * T^2 * KL(teacher_soft || student_soft)
      + (1 - alpha) * CE(student_logits, hard_labels)

where:
    student_soft = softmax(student_logits / T)
    teacher_soft = [1 - teacher_p, teacher_p]   (2-class soft label)
    T  = distillation temperature (default 2.0)
    alpha = KD weight (default 0.7)

KD is averaged over samples with valid teacher probabilities; CE is averaged
over the full batch. If every teacher probability is missing, KD is zero and
the total loss remains (1-alpha) * CE. Set alpha=0 for hard-label-only training.

Usage:
    python -m distil_vp.train --config configs/distil_default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from distil_vp.dataset import PhishingDistilDataset, collate_fn, load_split_df
from distil_vp.evaluate import compute_metrics, format_metrics
from distil_vp.model import DistilKoBERTClassifier


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def kd_loss(
    student_logits: torch.Tensor,   # (B, 2)
    teacher_p: torch.Tensor,        # (B,) soft probability for class 1
    temperature: float,
    mask: torch.Tensor,             # (B,) bool – True where teacher_p is valid
) -> torch.Tensor:
    """KL divergence between soft teacher and soft student distributions."""
    if not mask.any():
        return student_logits.sum() * 0.0  # keeps grad graph alive

    # Teacher soft labels: [[1-p, p], ...]
    tp = teacher_p[mask].unsqueeze(1)                          # (N, 1)
    teacher_soft = torch.cat([1.0 - tp, tp], dim=1)           # (N, 2)
    teacher_log_soft = torch.log(teacher_soft.clamp(min=1e-8))

    student_soft = F.log_softmax(student_logits[mask] / temperature, dim=-1)  # (N, 2)

    # F.kl_div expects (input=log_probs, target=probs)
    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    return loss * (temperature ** 2)


def total_loss(
    student_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    teacher_p: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    valid_mask = ~torch.isnan(teacher_p)
    ce = F.cross_entropy(student_logits, hard_labels)
    kd = kd_loss(student_logits, teacher_p, temperature, valid_mask)
    return alpha * kd + (1.0 - alpha) * ce


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@torch.no_grad()
def evaluate(model: DistilKoBERTClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().tolist()

        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds = (torch.argmax(logits, dim=-1)).cpu().tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

    return compute_metrics(all_labels, all_preds, all_probs)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict[str, Any]) -> None:
    seed = int(_get(cfg, "run", "seed", default=42))
    set_seed(seed)

    # ---- Data ----
    train_csv = _get(cfg, "data", "train_csv")
    text_col = _get(cfg, "data", "text_col", default="text")
    label_col = _get(cfg, "data", "label_col", default="label")
    teacher_p_col = _get(cfg, "data", "teacher_p_col", default="teacher_p")
    val_fraction = float(_get(cfg, "data", "val_fraction", default=0.15))

    train_df, val_df = load_split_df(train_csv, val_fraction=val_fraction, seed=seed)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    # ---- Tokenizer ----
    student_hf_id = _get(cfg, "student", "hf_id", default="monologg/distilkobert")
    max_length = int(_get(cfg, "student", "max_length", default=512))
    tokenizer = AutoTokenizer.from_pretrained(student_hf_id, trust_remote_code=True)

    train_dataset = PhishingDistilDataset(
        train_df, tokenizer, text_col=text_col, label_col=label_col,
        max_length=max_length, teacher_p_col=teacher_p_col,
    )
    val_dataset = PhishingDistilDataset(
        val_df, tokenizer, text_col=text_col, label_col=label_col,
        max_length=max_length, teacher_p_col=teacher_p_col,
    )

    batch_size = int(_get(cfg, "train", "batch_size", default=16))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = float(_get(cfg, "student", "dropout", default=0.1))
    freeze_encoder = bool(_get(cfg, "student", "freeze_encoder", default=False))
    model = DistilKoBERTClassifier(num_labels=2, dropout=dropout, freeze_encoder=freeze_encoder)
    model.to(device)

    # ---- Optimizer / Scheduler ----
    lr = float(_get(cfg, "train", "lr", default=3e-5))
    weight_decay = float(_get(cfg, "train", "weight_decay", default=0.01))
    num_epochs = int(_get(cfg, "train", "epochs", default=10))
    warmup_fraction = float(_get(cfg, "train", "warmup_fraction", default=0.1))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- KD hyperparams ----
    alpha = float(_get(cfg, "distill", "alpha", default=0.7))
    temperature = float(_get(cfg, "distill", "temperature", default=2.0))

    # ---- Output dir ----
    out_dir = Path(_get(cfg, "run", "out_dir", default="outputs/distil_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_ckpt = str(out_dir / "best_student.pt")
    history: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            tp = batch["teacher_p"].to(device)

            logits = model(ids, att)
            loss = total_loss(logits, labels, tp, alpha=alpha, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        row = {"epoch": epoch, "loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  {format_metrics(val_metrics)}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save(best_ckpt)
            print(f"  ↑ new best F1={best_val_f1:.4f}, saved → {best_ckpt}")

    # Save training history
    history_path = out_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete. Best val F1={best_val_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"History: {history_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distillation training")
    parser.add_argument("--config", required=True, help="Path to distil config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    train(cfg)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
