# Real PyTorch Backend

The simulator remains the default backend so the repo stays runnable without third-party packages. For a real evidence pass on a GPU-capable machine, use the torch backend.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-torch.txt
```

## Run

Compare both controllers on the real backend:

```bash
python3 -m catuskoti_ar.cli compare \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 30 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --output artifacts/torch_compare
```

Run a single controller:

```bash
python3 -m catuskoti_ar.cli run-loop \
  --controller vec3 \
  --backend torch \
  --data-dir data \
  --device cuda \
  --epochs 30 \
  --output artifacts/torch_vec3
```

## Notes

- The backend trains `torchvision.models.resnet18(weights=None, num_classes=100)`.
- `stochastic_depth_*` now applies residual drop-path across the ResNet blocks on the torch path.
- `dropout_high` remains an explicit classifier-dropout stress action.
- `eval_tta` changes the evaluation hash and should drive comparability negative as intended.
- The adapter logs real metrics for validation accuracy, train/val loss gap, gradient statistics, weight distance, and proxy metrics.
