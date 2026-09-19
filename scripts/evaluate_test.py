"""Evaluate training runs on the test split: held-out signs performed by held-out signers.

Prints each run's metrics with 95% bootstrap CIs over signs, and each run's gain over the first run,
paired over signs (mean per-sign difference in top-1, with a bootstrap CI). Twin signs (the prepared
set's twins.csv) are not scored against each other. The test split is for final numbers and important
comparisons; model decisions are made on validation.

With `--prepared` and `--name` it evaluates another prepared set of held-out signs the same way,
e.g. the Slovo cross-language evaluation set. `--signs` then matches its vocabulary size to another
set's, so that the identification metrics can be compared.

Run from the repo root: uv run scripts/evaluate_test.py gru_hide_low gru_auslan
Writes outputs/runs/<run>/<name>_summary.parquet and <name>_per_sign.parquet.
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
SUBSET_DRAWS = 5  # sign subsets drawn for --signs


def evaluate_set(embeddings: np.ndarray, clips: pl.DataFrame, signs: int | None, twins: set[tuple[str, str]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate all the clips, or, with `signs`, random subsets of that many signs, averaged over draws.

    A query is scored against every sign of the set, so top-1 and top-5 drop as the set holds more
    signs, and comparing them between two evaluation sets means matching their vocabulary sizes (see
    ROADMAP.md). The subsets are drawn the same way for every run.
    """
    if not signs:
        return evaluate(cosine_similarity(embeddings), clips, twins=twins)
    rng = np.random.default_rng(0)
    all_signs = clips["sign"].unique().sort().to_numpy()
    draws = []
    for _ in range(SUBSET_DRAWS):
        kept = clips["sign"].is_in(list(rng.choice(all_signs, signs, replace=False))).to_numpy()
        draws.append(evaluate(cosine_similarity(embeddings[kept]), clips.filter(kept), twins=twins))
    summary = pl.concat([summary for summary, _ in draws]).group_by("k", "metric", maintain_order=True).mean()
    per_sign = pl.concat([per_sign for _, per_sign in draws]).group_by("k", "sign", maintain_order=True).mean()
    return summary, per_sign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="runs to evaluate; gains are relative to the first")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--name", default="test", help="name of the evaluated set, used in the output file names")
    parser.add_argument("--signs", type=int, help=f"evaluate {SUBSET_DRAWS} random subsets of this many signs instead of the whole set")
    args = parser.parse_args()

    data = PreparedData(args.prepared)
    test = SignDataset(data, "test")
    clips = data.clips[test.positions]
    name = f"{args.name}_{args.signs}signs" if args.signs else args.name
    print(f"{name}: {len(clips)} clips, {clips['sign'].n_unique()} signs, {clips['signer'].n_unique()} signers")
    if args.signs:
        print(f"metrics averaged over {SUBSET_DRAWS} random subsets of {args.signs} signs")

    per_sign = {}
    for run in args.runs:
        _, model = load_run(RUNS_DIR / run, data)
        summary, per_sign[run] = evaluate_set(embed(model.to("cuda"), test, "cuda"), clips, args.signs, data.twins)
        summary.write_parquet(RUNS_DIR / run / f"{name}_summary.parquet")
        per_sign[run].write_parquet(RUNS_DIR / run / f"{name}_per_sign.parquet")
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
