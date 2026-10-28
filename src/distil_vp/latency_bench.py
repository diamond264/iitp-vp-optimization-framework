"""
Latency benchmark for all model variants.

Measures single-sample inference latency (ms) with:
  - GPU warm-up (5 samples discarded)
  - CUDA event timing for accurate GPU measurement
  - 100 timed repetitions, reporting mean ± std

LLM settings:  original, int8, int4, prune_10/30/50/80
Student settings: baseline (pretrained), hard_labels, kd

Usage:
    python -m distil_vp.latency_bench \
        --model_name   private_models/teacher \
        --test_csv     /path/to/test.csv \
        --kd_ckpt      outputs/distil_run_v2/best_student.pt \
        --hard_ckpt    outputs/hard_only_run_v2/best_student.pt \
        --out_json     outputs/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel,
)

from distil_vp.llm_compare import (
    build_prompt,
    load_original,
    load_int8,
    load_int4,
    load_pruned,
    TOKEN_ID_0,
    TOKEN_ID_1,
    MAX_INPUT_TOKENS,
)
from distil_vp.model import DistilKoBERTClassifier

STUDENT_HF_ID = "monologg/distilkobert"
WARMUP = 5
N_SAMPLES = 100


# ── GPU timing helper ─────────────────────────────────────────────────────────

def cuda_time_ms(fn) -> float:
    """Return elapsed GPU time of fn() in milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ── LLM single-sample benchmark ───────────────────────────────────────────────

@torch.no_grad()
def bench_llm(model, tokenizer, prompts: list[str], device) -> dict[str, float]:
    all_prompts = prompts[:WARMUP + N_SAMPLES]

    # warm-up
    for p in all_prompts[:WARMUP]:
        enc = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_TOKENS).to(device)
        enc.pop("token_type_ids", None)
        model(**enc)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for p in all_prompts[WARMUP:WARMUP + N_SAMPLES]:
        enc = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_TOKENS).to(device)
        enc.pop("token_type_ids", None)
        t = cuda_time_ms(lambda: model(**enc))
        times.append(t)

    return {"mean_ms": round(statistics.mean(times), 2),
            "std_ms":  round(statistics.stdev(times), 2),
            "min_ms":  round(min(times), 2),
            "max_ms":  round(max(times), 2)}


# ── DistilKoBERT single-sample benchmark ─────────────────────────────────────

@torch.no_grad()
def bench_student(model, tokenizer, texts: list[str], device) -> dict[str, float]:
    all_texts = texts[:WARMUP + N_SAMPLES]

    # warm-up
    for t in all_texts[:WARMUP]:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=512, padding=False).to(device)
        enc.pop("token_type_ids", None)
        model(**enc)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for t in all_texts[WARMUP:WARMUP + N_SAMPLES]:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=512, padding=False).to(device)
        enc.pop("token_type_ids", None)
        ms = cuda_time_ms(lambda: model(**enc))
        times.append(ms)

    return {"mean_ms": round(statistics.mean(times), 2),
            "std_ms":  round(statistics.stdev(times), 2),
            "min_ms":  round(min(times), 2),
            "max_ms":  round(max(times), 2)}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--test_csv",   required=True)
    parser.add_argument("--kd_ckpt",    required=True)
    parser.add_argument("--hard_ckpt",  required=True)
    parser.add_argument("--out_json",   default="outputs/latency.json")
    parser.add_argument(
        "--llm_settings", nargs="+",
        default=["original", "int8", "int4",
                 "prune_10", "prune_30", "prune_50", "prune_80"],
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    df = pd.read_csv(args.test_csv)
    texts   = df["text"].fillna("").astype(str).tolist()
    prompts = [build_prompt(t) for t in texts]

    results: dict[str, Any] = {}

    PRUNE_MAP = {
        "prune_10": 0.10, "prune_30": 0.30,
        "prune_50": 0.50, "prune_80": 0.80,
    }

    # ── LLM variants ──────────────────────────────────────────────────────────
    for setting in args.llm_settings:
        print(f"\n{'='*50}\nLLM setting: {setting}")
        try:
            if setting == "original":
                model, tok = load_original(args.model_name)
            elif setting == "int8":
                model, tok = load_int8(args.model_name)
            elif setting == "int4":
                model, tok = load_int4(args.model_name)
            elif setting in PRUNE_MAP:
                model, tok, *_ = load_pruned(args.model_name, PRUNE_MAP[setting])
            else:
                continue

            dev = next(iter(model.parameters())).device
            stats = bench_llm(model, tok, prompts, dev)
            results[f"llm_{setting}"] = stats
            print(f"  {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms/sample  "
                  f"(min={stats['min_ms']:.1f}, max={stats['max_ms']:.1f})")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[f"llm_{setting}"] = {"error": str(exc)}
        finally:
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

    # ── DistilKoBERT variants ─────────────────────────────────────────────────
    print(f"\n{'='*50}\nLoading DistilKoBERT tokenizer …")
    tok_s = AutoTokenizer.from_pretrained(
        STUDENT_HF_ID, trust_remote_code=True
    )

    student_configs = [
        ("student_baseline", None),
        ("student_hard",     args.hard_ckpt),
        ("student_kd",       args.kd_ckpt),
    ]

    for name, ckpt_path in student_configs:
        print(f"\n{'='*50}\nStudent setting: {name}")
        try:
            model_s = DistilKoBERTClassifier().to(device)
            if ckpt_path:
                state = torch.load(ckpt_path, map_location=device)
                model_s.load_state_dict(state)
            model_s.eval()

            stats = bench_student(model_s, tok_s, texts, device)
            results[name] = stats
            print(f"  {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms/sample  "
                  f"(min={stats['min_ms']:.1f}, max={stats['max_ms']:.1f})")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[name] = {"error": str(exc)}
        finally:
            try:
                del model_s
            except NameError:
                pass
            torch.cuda.empty_cache()

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Setting':<26}{'Mean (ms)':>12}{'Std (ms)':>10}{'Min (ms)':>10}{'Max (ms)':>10}")
    print("-" * 68)
    for k, v in results.items():
        if "error" in v:
            print(f"{k:<26}  ERROR")
        else:
            print(f"{k:<26}{v['mean_ms']:>12.1f}{v['std_ms']:>10.1f}"
                  f"{v['min_ms']:>10.1f}{v['max_ms']:>10.1f}")
    print("=" * 68)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
