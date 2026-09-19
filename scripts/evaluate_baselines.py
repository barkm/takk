"""Evaluate the baselines without training on a split of the prepared ASL Citizen data.

Run from the repo root: uv run scripts/evaluate_baselines.py [--split val]
Results (summary and per-sign metrics) are written to outputs/results/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.baselines import dtw_distances, dtw_features, hand_embedding
from isolated_sign_validation.evaluation import cosine_similarity, evaluate
from isolated_sign_validation.preparation import PrepConfig, PreparedData

OUT_DIR = Path("outputs/results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    data = PreparedData(args.prepared)
    positions = np.flatnonzero((data.clips["split"] == args.split).to_numpy())
    clips = data.clips[positions]
    print(f"{args.split}: {len(positions)} clips, {clips['sign'].n_unique()} signs, {clips['signer'].n_unique()} signers")

    def hand_features() -> np.ndarray:
        embeddings = np.stack([hand_embedding(data[i], data.config) for i in positions])
        return cosine_similarity(embeddings - embeddings.mean(axis=0))

    def dtw() -> np.ndarray:
        return -dtw_distances(np.stack([dtw_features(data[i], data.config) for i in positions]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, similarity in [("hand_features", hand_features), ("dtw", dtw)]:
        start = time.time()
        summary, per_sign = evaluate(similarity(), clips, twins=data.twins)
        summary.write_parquet(OUT_DIR / f"{name}_{args.split}_summary.parquet")
        per_sign.write_parquet(OUT_DIR / f"{name}_{args.split}_per_sign.parquet")
        table = summary.with_columns(
            pl.format("{} [{}, {}]", *(pl.col(c).round(3) for c in ("value", "ci_low", "ci_high"))).alias("estimate")
        ).pivot(on="metric", index="k", values="estimate")
        print(f"\n== {name} ({time.time() - start:.0f}s)\n{table}")
        print("hardest signs (k=5, by AUC):", per_sign.filter(pl.col("k") == 5).sort("auc").head(5)["sign"].to_list())


if __name__ == "__main__":
    main()
