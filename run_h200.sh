#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"

ARGS=(
  --data-dir "$SCRIPT_DIR/data"
  --log-dir "$SCRIPT_DIR/runs"
  --device auto
  --precision bf16
  --batch-size 32
  --epochs 100
  --val-interval "$VAL_INTERVAL"
)

if [[ "$COMPILE" == "1" ]]; then
  ARGS+=(--compile --compile-mode "$COMPILE_MODE")
fi

python "$SCRIPT_DIR/train_predictor.py" "${ARGS[@]}" "$@"
