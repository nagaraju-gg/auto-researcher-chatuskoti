from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from catuskoti_ar.config import ExperimentConfig, LoopConfig
from catuskoti_ar.runner import run_comparison, run_failure_injection_set, run_single_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Catuskoti AutoResearcher V1 prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Run both controllers and generate a comparison report")
    compare.add_argument("--output", type=Path, default=Path("artifacts/demo_run"))
    add_backend_args(compare)

    run_loop = subparsers.add_parser("run-loop", help="Run a single controller")
    run_loop.add_argument("--controller", choices=["vec3", "binary"], required=True)
    run_loop.add_argument("--output", type=Path, default=Path("artifacts/single_run"))
    add_backend_args(run_loop)

    failure_set = subparsers.add_parser("run-failure-set", help="Run the named failure injection set and generate a report")
    failure_set.add_argument("--output", type=Path, default=Path("artifacts/failure_set"))
    add_backend_args(failure_set)

    return parser


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["simulator", "torch"], default="simulator")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mode", choices=["default", "calibration", "challenge"], default="default")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    torch_cfg = replace(
        cfg.torch,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    return replace(cfg, backend=args.backend, torch=torch_cfg)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config(args)

    try:
        if args.command == "compare":
            run_comparison(args.output, cfg, iterations=args.iterations, seeds=args.seeds, mode=args.mode)
            print(f"Wrote comparison artifacts to {args.output}")
            return
        if args.command == "run-loop":
            result = run_single_loop(
                cfg,
                LoopConfig(controller=args.controller, max_iterations=args.iterations, n_seeds=args.seeds, mode=args.mode),
                args.output,
            )
            print(f"Wrote {args.controller} artifacts to {result.output_dir}")
            return
        if args.command == "run-failure-set":
            results = run_failure_injection_set(args.output, cfg, seeds=args.seeds)
            print(f"Wrote failure injection artifacts to {args.output} ({len(results)} cases)")
            return
    except ImportError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
