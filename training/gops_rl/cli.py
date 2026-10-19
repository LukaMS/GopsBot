from __future__ import annotations

import argparse
from pathlib import Path

from .arena import evaluate_run
from .export import export_release
from .ppo import resolve_device, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gops-rl", description="Train and release GOPS self-play policies")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train a masked PPO policy in a self-play league")
    train_parser.add_argument("--config", type=Path, default=Path("training/configs/ppo.yaml"))
    train_parser.add_argument("--run-dir", type=Path, default=Path("training/runs/latest"))
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--resume", type=Path)

    evaluation_parser = commands.add_parser("evaluate", help="Evaluate and rank policy checkpoints")
    evaluation_parser.add_argument("--run-dir", type=Path, required=True)
    evaluation_parser.add_argument("--games", type=int, default=10000)
    evaluation_parser.add_argument("--seed", type=int, default=9182)
    evaluation_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluation_parser.add_argument("--updates", type=int, nargs="+", help="Evaluate specific checkpoint update numbers")

    export_parser = commands.add_parser("export", help="Export selected policies to browser-ready ONNX")
    export_parser.add_argument("--run-dir", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, default=Path("web/public/models"))
    export_parser.add_argument("--allow-uncalibrated", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        result = train(args.config, args.run_dir, args.device, args.resume)
    elif args.command == "evaluate":
        result = evaluate_run(
            args.run_dir,
            args.games,
            resolve_device(args.device),
            args.seed,
            set(args.updates) if args.updates else None,
        )
    else:
        result = export_release(args.run_dir, args.output, args.allow_uncalibrated)
    print(result)
