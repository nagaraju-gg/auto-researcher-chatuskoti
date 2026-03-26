#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m catuskoti_ar.cli compare \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --iterations 4 \
  --seeds 3 \
  --output artifacts/torch_compare
