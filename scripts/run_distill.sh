#!/usr/bin/env bash
# Step 2: Train the DistilKoBERT student with knowledge distillation.
# Make sure you have run extract_teacher_labels.sh first and updated the
# train_csv path in the config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CFG="${1:-$ROOT_DIR/configs/distil_default.yaml}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT_DIR/src/distil_vp/train.py" --config "$CFG"
