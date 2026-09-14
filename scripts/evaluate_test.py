"""Evaluate training runs on the test split: held-out signs performed by held-out signers.

Prints each run's metrics with 95% bootstrap CIs over signs, and each run's gain over the first run,
paired over signs (mean per-sign difference in top-1, with a bootstrap CI). The test split is for final
numbers and important comparisons; model decisions are made on validation.

Run from the repo root: uv run scripts/evaluate_test.py gru_hide_low gru_auslan
Writes outputs/runs/<run>/test_summary.parquet and test_per_sign.parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, evaluate
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
METRICS = ["auc", "eer", "top1", "top5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="runs to evaluate; gains are relative to the first")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    args = parser.parse_args()

    data = PreparedData(args.prepared)
    test = SignDataset(data, "test")
    clips = data.clips[test.positions]
    print(f"test: {len(clips)} clips, {clips['sign'].n_unique()} signs, {clips['signer'].n_unique()} signers")

    per_sign = {}
    for run in args.runs:
        _, model = load_run(RUNS_DIR / run, data)
        summary, per_sign[run] = evaluate(cosine_similarity(embed(model.to("cuda"), test, "cuda")), clips)
        summary.write_parquet(RUNS_DIR / run / "test_summary.parquet")
        per_sign[run].write_parquet(RUNS_DIR / run / "test_per_sign.parquet")
        print(f"\n{run}:")
        for k, group in summary.group_by("k", maintain_order=True):
            values = {m: f"{v:.4f} [{low:.4f}, {high:.4f}]" for m, v, low, high in group.select("metric", "value", "ci_low", "ci_high").iter_rows()}
            print(f"  k={k[0]}: " + "  ".join(f"{m} {values[m]}" for m in METRICS))

    rng = np.random.default_rng(0)
    base = args.runs[0]
    for run in args.runs[1:]:
        print(f"\n{run} vs {base}, per-sign top-1 gain (mean over signs, 95% bootstrap CI):")
        joined = per_sign[base].join(per_sign[run], on=["k", "sign"], suffix="_run")
        for k, group in joined.group_by("k", maintain_order=True):
            gain = (group["top1_run"] - group["top1"]).to_numpy()
            boot = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(2000)]
            print(f"  k={k[0]}: {gain.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
