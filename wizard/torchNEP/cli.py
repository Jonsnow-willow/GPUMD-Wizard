from __future__ import annotations

import argparse
from pathlib import Path

from wizard.torchNEP.train import train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchnep", description="TorchNEP training tools.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a run directory containing nep.in.")
    train_parser.add_argument("run_dir", nargs="?", default=".", type=Path, help="Training run directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "train"):
        train_run(getattr(args, "run_dir", Path(".")))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
