"""
LLM compression comparison: evaluate HyperCLOVA-X 0.5B on test.csv under
several compression settings and compare with the distilled DistilKoBERT.

Settings:
  original   – full fp16 (24 layers, ~500 M params)
  int8       – bitsandbytes INT8 quantization
  int4       – bitsandbytes NF4 4-bit quantization
  prune_10   – structural pruning: drop 10 % of layers  (~2  layers removed)
  prune_30   – structural pruning: drop 30 % of layers  (~7  layers removed)
  prune_50   – structural pruning: drop 50 % of layers  (~12 layers removed)
  prune_80   – structural pruning: drop 80 % of layers  (~19 layers removed)

Structural pruning removes transformer blocks at uniformly-spaced indices,
keeping the first and last layers intact where possible.

For every setting the script:
  1. Loads (and optionally modifies) the model.
  2. Extracts logits at the last prompt position for answer tokens "0" (id=15)
     and "1" (id=16) to obtain P(phishing) — enabling both hard prediction and
     AUC computation without any autoregressive generation.
  3. Reports accuracy, F1, precision, recall, and AUC on the full test CSV.

Usage:
    python -m distil_vp.llm_compare \\
        --model_name private_models/teacher \\
        --test_csv   /path/to/test.csv \\
        --out_json   outputs/llm_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distil_vp.evaluate import compute_metrics

# ── prompt ────────────────────────────────────────────────────────────────────

INSTRUCTION = (
    "통화 전사/요약에서 정부기관(검·경·금감원 등) 또는 금융기관을 사칭해 범죄 연루·수사 협조를 요구하거나, "
    "개인정보/송금·현금인출/앱 설치·원격제어/OTP·인증번호 요구 및 긴급·협박을 동반하면 1(보이스피싱), 아니면 0(정상).\n"
    "출력은 설명 없이 0 또는 1 한 글자만 반환한다."
)
TOKEN_ID_0 = 15
TOKEN_ID_1 = 16
MAX_INPUT_TOKENS = 2048


def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n문장:{text}\n정답: "


# ── model loaders ─────────────────────────────────────────────────────────────

def _base_kwargs(trust: bool = True) -> dict[str, Any]:
    return {"trust_remote_code": trust}


def load_original(model_name: str) -> tuple:
    """Full fp16 model."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, **_base_kwargs()
    )
    model.to("cuda").eval()
    return model, tok


def load_int8(model_name: str) -> tuple:
    """bitsandbytes INT8 quantization (LLM.int8())."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=cfg, device_map="auto", **_base_kwargs()
    )
    model.eval()
    return model, tok


def load_int4(model_name: str) -> tuple:
    """bitsandbytes NF4 4-bit quantization."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=cfg, device_map="auto", **_base_kwargs()
    )
    model.eval()
    return model, tok


def _uniform_remove_indices(n_total: int, n_remove: int) -> list[int]:
    """Pick n_remove indices spread uniformly across [0, n_total-1].
    Avoids removing the very first and last layers where possible.
    """
    if n_remove == 0:
        return []
    # Work on the interior layers first; if we must touch ends, we do so.
    interior = list(range(1, n_total - 1))
    if n_remove <= len(interior):
        step = len(interior) / n_remove
        return sorted(int(interior[int(i * step + step / 2)]) for i in range(n_remove))
    # Need to also remove boundary layers
    all_idx = list(range(n_total))
    step = n_total / n_remove
    return sorted(int(i * step + step / 2) for i in range(n_remove))


def load_pruned(model_name: str, prune_ratio: float) -> tuple:
    """Load fp16 model and drop a fraction of transformer layers uniformly."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, **_base_kwargs()
    )

    layers: nn.ModuleList = model.model.layers  # type: ignore[attr-defined]
    n_total = len(layers)
    n_remove = round(n_total * prune_ratio)
    remove_set = set(_uniform_remove_indices(n_total, n_remove))
    kept = [layer for i, layer in enumerate(layers) if i not in remove_set]

    model.model.layers = nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)

    model.to("cuda").eval()
    return model, tok, n_total, len(kept), sorted(remove_set)


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
    desc: str = "",
) -> tuple[list[int], list[float]]:
    """Return (hard_predictions, phishing_probabilities)."""
    prompts = [build_prompt(t) for t in texts]
    preds: list[int] = []
    probs: list[float] = []
    device = next(iter(model.parameters())).device

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc or "Eval", leave=False):
        batch = prompts[start: start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits          # (B, seq_len, vocab)
        last = logits[:, -1, :].float()       # (B, vocab)
        answer = last[:, [TOKEN_ID_0, TOKEN_ID_1]]   # (B, 2)
        p = torch.softmax(answer, dim=-1)[:, 1].cpu().tolist()
        pred = [1 if pi >= 0.5 else 0 for pi in p]
        probs.extend(p)
        preds.extend(pred)

    return preds, probs


# ── model size helper ─────────────────────────────────────────────────────────

def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--test_csv",   required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_json",   default="outputs/llm_comparison.json")
    parser.add_argument(
        "--settings", nargs="+",
        default=["original", "int8", "int4",
                 "prune_10", "prune_30", "prune_50", "prune_80"],
        help="Which settings to run (space-separated subset of the default list)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.test_csv)
    texts  = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].fillna(0).astype(int).tolist()
    print(f"Test samples: {len(df)}  (label=0: {labels.count(0)}, label=1: {labels.count(1)})\n")

    results: dict[str, dict] = {}
    PRUNE_MAP = {
        "prune_10": 0.10,
        "prune_30": 0.30,
        "prune_50": 0.50,
        "prune_80": 0.80,
    }

    for setting in args.settings:
        print(f"{'='*60}")
        print(f"Setting: {setting}")
        t0 = time.time()

        extra: dict[str, Any] = {}
        try:
            if setting == "original":
                model, tok = load_original(args.model_name)
                extra["layers_kept"] = model.config.num_hidden_layers
                extra["params_M"] = round(param_count(model) / 1e6, 1)

            elif setting == "int8":
                model, tok = load_int8(args.model_name)
                extra["layers_kept"] = model.config.num_hidden_layers
                extra["params_M"] = round(param_count(model) / 1e6, 1)

            elif setting == "int4":
                model, tok = load_int4(args.model_name)
                extra["layers_kept"] = model.config.num_hidden_layers
                extra["params_M"] = round(param_count(model) / 1e6, 1)

            elif setting in PRUNE_MAP:
                ratio = PRUNE_MAP[setting]
                model, tok, n_orig, n_kept, removed = load_pruned(args.model_name, ratio)
                extra["layers_total"] = n_orig
                extra["layers_removed"] = len(removed)
                extra["layers_kept"] = n_kept
                extra["removed_indices"] = removed
                extra["params_M"] = round(param_count(model) / 1e6, 1)
                print(f"  Removed {len(removed)}/{n_orig} layers → {n_kept} layers remain")
                print(f"  Removed indices: {removed}")

            else:
                print(f"  Unknown setting, skipping.")
                continue

            preds, probs = run_eval(model, tok, texts,
                                    batch_size=args.batch_size, desc=setting)
            metrics = compute_metrics(labels, preds, probs)
            elapsed = round(time.time() - t0, 1)

            row = {**extra, **metrics, "elapsed_s": elapsed}
            results[setting] = row
            print(f"  params={extra.get('params_M','?')} M  "
                  f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
                  f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
                  f"auc={metrics['auc']:.4f}  ({elapsed}s)")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[setting] = {"error": str(exc)}

        finally:
            # Free GPU memory between runs
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

    # ── print table ───────────────────────────────────────────────────────────
    cols = ["accuracy", "f1", "precision", "recall", "auc"]
    hdr = f"\n{'Setting':<14}{'Layers':>8}{'Params M':>10}" + \
          "".join(f"{c:>11}" for c in cols)
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<14}  ERROR: {r['error']}")
            continue
        vals = "".join(f"{r.get(c, float('nan')):>11.4f}" for c in cols)
        print(f"{name:<14}{r.get('layers_kept', '?'):>8}{r.get('params_M', '?'):>10}{vals}")
    print("=" * len(hdr))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
