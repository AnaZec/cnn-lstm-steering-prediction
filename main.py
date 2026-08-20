"""Single entry point for training, evaluation and the presentation demo."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNN+LSTM steering-angle prediction project"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "evaluate", "demo", "all"],
        default="demo",
        help="Action to run. Default: demo",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Maximum validation sequences shown by the demo.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run demo without opening the OpenCV window; annotated frames are still saved.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=50,
        help="Milliseconds between demo frames.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command in ("train", "all"):
        from train import train
        train(args.config)

    if args.command in ("evaluate", "all"):
        from evaluate import evaluate
        evaluate(args.config)

    if args.command in ("demo", "all"):
        from demo import run_demo
        run_demo(
            args.config,
            max_sequences=args.frames,
            show_window=not args.no_window,
            delay_ms=args.delay,
        )


if __name__ == "__main__":
    main()
