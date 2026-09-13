"""Train a sign embedding model on the prepared ASL Citizen data.

Run from the repo root:
    uv run scripts/train.py --name gru_arcface
    uv run scripts/train.py --name gru_dropout --set dropout=0.4 --set augment.hand_drop=0.2
Settings are the fields of TrainConfig (and AugmentConfig as augment.<field>).
Writes outputs/runs/<name>/ (config, metrics.csv, curves.png, best.pt, validation evaluation).
"""

import argparse
import ast
import dataclasses
from pathlib import Path

from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import TrainConfig, train


def with_override(config, key: str, value: str):
    """`config` with the (possibly nested, dot-separated) field `key` set to `value`: taken as is for
    text fields, otherwise parsed as a Python literal."""
    field, _, rest = key.partition(".")
    current = getattr(config, field)
    if rest:
        return dataclasses.replace(config, **{field: with_override(current, rest, value)})
    if isinstance(current, str):
        return dataclasses.replace(config, **{field: value})
    parsed = ast.literal_eval(value)
    return dataclasses.replace(config, **{field: tuple(parsed) if isinstance(current, tuple) else type(current)(parsed)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a training setting")
    args = parser.parse_args()

    config = TrainConfig()
    for override in args.set:
        key, _, value = override.partition("=")
        config = with_override(config, key, value)
    run_dir = Path("outputs/runs") / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose another --name")
    train(config, PreparedData(args.prepared), run_dir)


if __name__ == "__main__":
    main()
