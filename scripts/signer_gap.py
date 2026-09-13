"""Measure how well a model generalizes to unseen signers.

Compares a run trained without some training signers (--held-out, see TrainConfig.holdout_signers)
with the same setup trained on all of them (--full), on the validation signs performed by the
held-out signers (unseen by the held-out run) and by the other signers (seen by both runs). The
held-out run's drop on the other signers only reflects its smaller training set; the signer gap is
its additional drop on the held-out signers.

Run from the repo root: uv run scripts/signer_gap.py --held-out gru_holdout7 --full gru_reg_aug
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, evaluate
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
METRICS = ["eer@1", "top1@1", "eer@5", "top1@5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", required=True, help="run trained with holdout_signers")
    parser.add_argument("--full", required=True, help="the same setup trained on all training signers")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    args = parser.parse_args()

    data = PreparedData(args.prepared)
    val = SignDataset(data, "val")
    clips = data.clips[val.positions]
    held_out_config, _ = load_run(RUNS_DIR / args.held_out, data)
    unseen = clips["signer"].is_in(list(held_out_config.holdout_signers)).to_numpy()
    print(f"{unseen.sum()} val clips by {len(held_out_config.holdout_signers)} held-out signers, {(~unseen).sum()} by the others")

    rows = []
    for run in (args.full, args.held_out):
        _, model = load_run(RUNS_DIR / run, data)
        similarity = cosine_similarity(embed(model.to("cuda"), val, "cuda"))
        for group, queries in [("held-out signers", unseen), ("other signers", ~unseen)]:
            summary, _ = evaluate(similarity, clips, ks=(1, 5), queries=queries)
            values = {f"{m}@{k}": (v, low, high) for k, m, v, low, high in summary.select("k", "metric", "value", "ci_low", "ci_high").iter_rows()}
            rows.append({"run": run, "queries": group} | {m: values[m] for m in METRICS})

    table = pl.DataFrame([{**r, **{m: f"{r[m][0]:.4f} [{r[m][1]:.4f}, {r[m][2]:.4f}]" for m in METRICS}} for r in rows])
    with pl.Config(tbl_width_chars=200):
        print(table)
    value = {(r["run"], r["queries"], m): r[m][0] for r in rows for m in METRICS}
    print("\nchange from the full run to the held-out run (held-out signers / other signers / signer gap):")
    for m in METRICS:
        drop_unseen = value[(args.held_out, "held-out signers", m)] - value[(args.full, "held-out signers", m)]
        drop_other = value[(args.held_out, "other signers", m)] - value[(args.full, "other signers", m)]
        print(f"  {m:7s} {drop_unseen:+.4f} / {drop_other:+.4f} / {drop_unseen - drop_other:+.4f}")


if __name__ == "__main__":
    main()
