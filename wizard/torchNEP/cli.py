from __future__ import annotations

import argparse
from pathlib import Path

from wizard.torchNEP.checkpoint import load_checkpoint_file
from wizard.torchNEP.evaluate import (
    compare_artifacts,
    evaluate_artifact,
    export_artifact,
    format_summary,
    require_existing,
    resolve_artifact_path,
)
from wizard.torchNEP.parser import load_train_config
from wizard.torchNEP.train import train_from_config, train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchnep", description="TorchNEP training tools.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a run directory containing nep.in.")
    train_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Training run directory.")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path, e.g. checkpoints/last.pt.")
    train_parser.add_argument("--device", type=str, default=None, help="Override device from nep.in.")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs from nep.in.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained artifact against train/test labels.")
    eval_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Run directory containing nep.in.")
    eval_parser.add_argument("--artifact", default="checkpoints/best.pt", help="Checkpoint or nep.txt to evaluate.")
    eval_parser.add_argument("--split", choices=("train", "test"), default=None, help="Dataset split. Defaults to test if available.")
    eval_parser.add_argument("--device", type=str, default=None, help="Override device from nep.in.")

    export_parser = subparsers.add_parser("export", help="Export a checkpoint to GPUMD-compatible nep.txt.")
    export_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Run directory containing nep.in.")
    export_parser.add_argument("--artifact", default="checkpoints/best.pt", help="Checkpoint or nep.txt to export.")
    export_parser.add_argument("--output", default="exports/nep.txt", help="Output nep.txt path, relative to run_dir by default.")
    export_parser.add_argument("--device", type=str, default=None, help="Override device from nep.in.")

    compare_parser = subparsers.add_parser("compare", help="Compare two artifacts on the same structures.")
    compare_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Run directory containing nep.in.")
    compare_parser.add_argument("--left", default="exports/nep.txt", help="Left artifact path.")
    compare_parser.add_argument("--right", default="checkpoints/best.pt", help="Right artifact path.")
    compare_parser.add_argument("--split", choices=("train", "test"), default=None, help="Dataset split. Defaults to test if available.")
    compare_parser.add_argument("--device", type=str, default=None, help="Override device from nep.in.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect checkpoint metadata.")
    inspect_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Run directory containing nep.in.")
    inspect_parser.add_argument("--artifact", default="checkpoints/best.pt", help="Checkpoint to inspect.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "train"):
        run_dir = getattr(args, "run_dir", Path("."))
        if args.command is None:
            train_run(run_dir)
            return
        config = load_train_config(run_dir)
        if args.resume is not None:
            config.runtime.resume = args.resume
        if args.device is not None:
            config.runtime.device = args.device
        if args.epochs is not None:
            config.optimizer.epochs = args.epochs
        train_from_config(config)
        return

    config = load_train_config(args.run_dir)
    if getattr(args, "device", None) is not None:
        config.runtime.device = args.device

    if args.command == "eval":
        print(format_summary(evaluate_artifact(config, artifact=args.artifact, split=args.split, device=args.device)))
        return

    if args.command == "export":
        output = export_artifact(config, artifact=args.artifact, output=args.output, device=args.device)
        print(f"Exported {output}")
        return

    if args.command == "compare":
        print(format_summary(compare_artifacts(config, left=args.left, right=args.right, split=args.split, device=args.device)))
        return

    if args.command == "inspect":
        inspect_checkpoint(config, args.artifact)
        return

    parser.error(f"Unknown command: {args.command}")


def inspect_checkpoint(config, artifact: str) -> None:
    path = require_existing(resolve_artifact_path(config, artifact))
    if path.suffix == ".txt":
        print(f"nep.txt artifact: {path}")
        return
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    print(f"Checkpoint: {path}")
    print(f"  epoch: {checkpoint.get('epoch')}")
    print(f"  best_metric: {checkpoint.get('best_metric')}")
    print(f"  has_optimizer: {checkpoint.get('optimizer_state_dict') is not None}")
    print(f"  has_scheduler: {checkpoint.get('scheduler_state_dict') is not None}")
    print(f"  has_rng_state: {checkpoint.get('rng_state') is not None}")
    if checkpoint.get("metrics"):
        print(f"  metrics: {checkpoint['metrics']}")


if __name__ == "__main__":
    main()
