"""Train a sign embedding model on the prepared ASL Citizen data.

Run from the repo root: uv run scripts/train.py --name gru_arcface [--epochs 60]
Writes outputs/runs/<name>/ (config, metrics.csv, curves.png, best.pt, validation evaluation).
"""

import argparse
from pathlib import Path

from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = parser.parse_args()

    run_dir = Path("outputs/runs") / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose another --name")
    train(TrainConfig(epochs=args.epochs), PreparedData(args.prepared), run_dir)


if __name__ == "__main__":
    main()
