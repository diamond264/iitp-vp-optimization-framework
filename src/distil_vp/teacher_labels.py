"""
Extract soft probability labels from the fine-tuned HyperCLOVA-X teacher model.

The teacher is a decoder-only LM prompted to output a single token: "0" (normal) or "1"
(voice phishing).  We extract the raw logits at the last prompt position for those two
answer tokens and convert them to a probability with a softmax, optionally temperature-
scaled.  The result is saved as an additional column ``teacher_p`` in a CSV file that
the distillation training loop can consume offline (no teacher needed at train time).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Token IDs for "0" and "1" in the HyperCLOVA-X tokenizer.
# These are fixed for this model family; verified empirically.
TOKEN_ID_0 = 15
TOKEN_ID_1 = 16

INSTRUCTION = (
    "통화 전사/요약에서 정부기관(검·경·금감원 등) 또는 금융기관을 사칭해 범죄 연루·수사 협조를 요구하거나, "
    "개인정보/송금·현금인출/앱 설치·원격제어/OTP·인증번호 요구 및 긴급·협박을 동반하면 1(보이스피싱), 아니면 0(정상).\n"
    "출력은 설명 없이 0 또는 1 한 글자만 반환한다."
)


def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n문장:{text}\n정답: "


def _resolve_dtype(s: str) -> torch.dtype:
    low = s.lower()
    if low in {"half", "fp16", "float16"}:
        return torch.float16
    if low in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def extract_teacher_labels(
    model_name: str,
    texts: list[str],
    batch_size: int = 8,
    max_input_tokens: int = 2048,
    dtype_str: str = "half",
    temperature: float = 1.0,
) -> list[float]:
    """Return per-sample P(phishing) from the teacher model.

    Args:
        model_name: Path or HF hub ID for the fine-tuned causal LM.
        texts: Raw conversation texts (no prompt wrapping needed).
        batch_size: Number of samples per forward pass.
        max_input_tokens: Max tokens to keep from each prompt (left-truncated).
        dtype_str: "half" | "bf16" | "fp32".
        temperature: Softmax temperature applied to the two answer logits.
            Higher T → softer labels; T=1.0 = raw softmax.

    Returns:
        List of floats in [0, 1] representing P(phishing) for each text.
    """
    dtype = _resolve_dtype(dtype_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = dtype if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=load_dtype
    )
    model.to(device)
    model.eval()

    prompts = [build_prompt(t) for t in texts]
    probs: list[float] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=load_dtype)
        if device.type == "cuda" and load_dtype in {torch.float16, torch.bfloat16}
        else torch.inference_mode()
    )

    for start in tqdm(range(0, len(prompts), batch_size), desc="Teacher labels"):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            if device.type == "cuda" and load_dtype in {torch.float16, torch.bfloat16}:
                with torch.autocast(device_type="cuda", dtype=load_dtype):
                    logits = model(**enc).logits  # (B, seq_len, vocab_size)
            else:
                logits = model(**enc).logits

        # Last non-padded position: use attention_mask to find it.
        # For left-padded inputs, the last token is always at position -1.
        last_logits = logits[:, -1, :]  # (B, vocab_size)
        answer_logits = last_logits[:, [TOKEN_ID_0, TOKEN_ID_1]].float()  # (B, 2)
        scaled = answer_logits / temperature
        p = torch.softmax(scaled, dim=-1)[:, 1]  # P(phishing) = P(token=="1")
        probs.extend(p.cpu().tolist())

    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract teacher soft labels")
    parser.add_argument("--config", required=True, help="Path to distil config YAML")
    parser.add_argument("--out_csv", default="", help="Override output CSV path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    teacher_name = cfg["teacher"]["model_name"]
    data_csv = cfg["data"]["train_csv"]
    text_col = cfg["data"].get("text_col", "text")
    label_col = cfg["data"].get("label_col", "label")
    batch_size = cfg["teacher"].get("batch_size", 4)
    max_input_tokens = cfg["teacher"].get("max_input_tokens", 2048)
    dtype_str = cfg["teacher"].get("dtype", "half")
    temperature = float(cfg["distill"].get("teacher_temperature", 1.0))

    df = pd.read_csv(data_csv)
    texts = df[text_col].fillna("").astype(str).tolist()

    probs = extract_teacher_labels(
        model_name=teacher_name,
        texts=texts,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        dtype_str=dtype_str,
        temperature=temperature,
    )

    df["teacher_p"] = probs

    out_csv = args.out_csv or str(
        Path(data_csv).with_name(Path(data_csv).stem + "_teacher_labels.csv")
    )
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved teacher labels → {out_csv}")
    print(f"  label=0 mean teacher_p: {df[df[label_col]==0]['teacher_p'].mean():.4f}")
    print(f"  label=1 mean teacher_p: {df[df[label_col]==1]['teacher_p'].mean():.4f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
