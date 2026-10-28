#!/usr/bin/env bash
# Step 1: Run this ONCE to extract teacher soft labels and save them to
#   private_data/train_teacher_labels.csv (never publish this file).
# Run from the repository root. The training configs already use this output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CFG="${1:-$ROOT_DIR/configs/teacher_labels.yaml}"
OUT_CSV="${2:-$ROOT_DIR/private_data/train_teacher_labels.csv}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT_DIR/src/distil_vp/teacher_labels.py" \
    --config "$CFG" \
    --out_csv "$OUT_CSV"

echo "Done. Update configs/distil_default.yaml → data.train_csv: $OUT_CSV"
