"""
Three-way comparison of DistilKoBERT on the voice phishing detection task.

Conditions evaluated on test.csv (held-out):
  1. Baseline  — randomly-initialized head, no training
  2. Hard-only — fine-tuned with CE on hard labels only (alpha=0)
  3. Distilled — trained with KL + CE using teacher soft labels (alpha=0.7, T=2)

Usage:
    # Evaluate on a separate test CSV (preferred):
    python -m distil_vp.compare \\
        --kd_ckpt   outputs/distil_run_v2/best_student.pt \\
        --hard_ckpt outputs/hard_only_run_v2/best_student.pt \\
        --test_csv  /path/to/test.csv

    # Or evaluate on the val split of a training CSV:
    python -m distil_vp.compare \\
        --kd_ckpt   outputs/distil_run/best_student.pt \\
        --hard_ckpt outputs/hard_only_run/best_student.pt \\
        --data_csv  /path/to/challenge_teacher_labels.csv \\
        --val_fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from distil_vp.dataset import PhishingDistilDataset, collate_fn, load_split_df
from distil_vp.evaluate import compute_metrics
from distil_vp.model import DistilKoBERTClassifier, STUDENT_HF_ID


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: DistilKoBERTClassifier,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> dict[str, float]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().tolist()
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_labels.extend(labels)
        all_preds.extend(probs)       # keep raw probs for threshold sweep
        all_probs.extend(probs)

    hard_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    metrics = compute_metrics(all_labels, hard_preds, all_probs)
    return metrics


# ── pretty table ──────────────────────────────────────────────────────────────

def print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    cols = ["accuracy", "f1", "precision", "recall", "auc"]
    header = f"{'Condition':<22}" + "".join(f"{c:>11}" for c in cols)
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in rows:
        vals = "".join(f"{m.get(c, float('nan')):>11.4f}" for c in cols)
        print(f"{name:<22}{vals}")
    print("=" * len(header))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kd_ckpt",      required=True,  help="KD checkpoint .pt")
    parser.add_argument("--hard_ckpt",    required=True,  help="Hard-label checkpoint .pt")
    # Provide either --test_csv (preferred holdout) or --data_csv + --val_fraction
    parser.add_argument("--test_csv",     default="",     help="Held-out test CSV (text, label columns)")
    parser.add_argument("--data_csv",     default="",     help="Training CSV; val split used if --test_csv absent")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--out_json",     default="",     help="Optional path to save results")
    args = parser.parse_args()

    if not args.test_csv and not args.data_csv:
        raise ValueError("Provide --test_csv or --data_csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    if args.test_csv:
        import pandas as pd
        val_df = pd.read_csv(args.test_csv)
        print(f"Test CSV: {args.test_csv}")
    else:
        _, val_df = load_split_df(args.data_csv, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Eval samples: {len(val_df)}  (label=0: {(val_df['label']==0).sum()}, label=1: {(val_df['label']==1).sum()})")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_HF_ID, trust_remote_code=True)
    val_dataset = PhishingDistilDataset(
        val_df, tokenizer, max_length=args.max_length, teacher_p_col="teacher_p"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    results: list[tuple[str, dict]] = []

    # ── 1. Baseline: untrained model ──────────────────────────────────────────
    print("\n[1/3] Evaluating baseline (untrained random head)...")
    baseline = DistilKoBERTClassifier(num_labels=2, dropout=0.0)
    baseline.to(device)
    m = evaluate_model(baseline, val_loader, device, "Baseline")
    results.append(("Baseline (untrained)", m))
    del baseline

    # ── 2. Hard-label fine-tuned ──────────────────────────────────────────────
    print("[2/3] Evaluating hard-label fine-tuned model...")
    hard_model = DistilKoBERTClassifier.load(
        args.hard_ckpt, map_location=device
    )
    hard_model.to(device)
    m = evaluate_model(hard_model, val_loader, device, "Hard-label")
    results.append(("Hard labels (CE only)", m))
    del hard_model

    # ── 3. Knowledge-distilled ────────────────────────────────────────────────
    print("[3/3] Evaluating knowledge-distilled model...")
    kd_model = DistilKoBERTClassifier.load(
        args.kd_ckpt, map_location=device
    )
    kd_model.to(device)
    m = evaluate_model(kd_model, val_loader, device, "KD")
    results.append(("KD (α=0.7, T=2)", m))
    del kd_model

    # ── print ─────────────────────────────────────────────────────────────────
    print_table(results)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({name: metrics for name, metrics in results}, f, indent=2)
        print(f"Saved results → {args.out_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
