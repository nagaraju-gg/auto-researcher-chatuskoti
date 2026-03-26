from __future__ import annotations

import platform
import sys


def main() -> int:
    print("Environment check for Catuskoti AutoResearcher torch backend")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
        import torchvision
    except ModuleNotFoundError as exc:
        print("")
        print("Missing dependency:")
        print(f"- {exc.name}")
        print("")
        print("Install with:")
        print("python3 -m pip install -r requirements-torch.txt")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    else:
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"MPS available: {mps_ok}")
        if not mps_ok:
            print("No GPU backend detected. CPU runs will work but be slow.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

