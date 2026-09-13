"""Compare training runs: their settings that differ from the defaults and their validation metrics.

Run from the repo root: uv run scripts/compare_runs.py
"""

import dataclasses
import json
from pathlib import Path

import polars as pl

from isolated_sign_validation.training import TrainConfig

RUNS_DIR = Path("outputs/runs")


def flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = tuple(value) if isinstance(value, list) else value
    return flat


def main() -> None:
    defaults = flatten(dataclasses.asdict(TrainConfig()))
    rows = []
    for run in sorted(RUNS_DIR.iterdir()):
        if not (run / "val_summary.parquet").exists():
            continue
        config = flatten(json.loads((run / "config.json").read_text()))
        changes = ", ".join(f"{k}={v}" for k, v in config.items() if defaults.get(k) != v)
        summary = pl.read_parquet(run / "val_summary.parquet")
        values = {f"{m}@{k}": v for k, m, v in summary.select("k", "metric", "value").iter_rows()}
        metrics = pl.read_csv(run / "metrics.csv")
        selection = "val_eer_k1" if "val_eer_k1" in metrics.columns else "val_auc"  # older runs selected on AUC
        best = metrics.drop_nulls(selection).sort(selection, descending=selection == "val_auc")["epoch"][0]
        rows.append({"run": run.name, "changes": changes, "best_epoch": best} | {key: values[key] for key in ("eer@1", "top1@1", "eer@5", "top1@5", "auc@5")})
    table = pl.DataFrame(rows).with_columns(pl.col("eer@1", "top1@1", "eer@5", "top1@5", "auc@5").round(4)).sort("eer@1")
    with pl.Config(tbl_rows=-1, tbl_width_chars=250, fmt_str_lengths=120):
        print(table)


if __name__ == "__main__":
    main()
