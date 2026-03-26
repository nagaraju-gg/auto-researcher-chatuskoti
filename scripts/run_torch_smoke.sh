#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m chatuskoti_evals.cli compare \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 1 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --iterations 1 \
  --seeds 1 \
  --output artifacts/torch_smoke

