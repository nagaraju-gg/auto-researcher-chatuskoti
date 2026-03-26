#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output artifacts/strong_v1/canonical_failure

python3 scripts/generate_failure_figure.py \
  artifacts/strong_v1/canonical_failure/failure_injection/failure_results.json

"$PYTHON_BIN" -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --iterations 4 \
  --seeds 3 \
  --output artifacts/strong_v1/challenge_compare

"$PYTHON_BIN" -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output artifacts/strong_v1/ablations
