"""Prepare the ASL Citizen clips of the train, val and test splits for training.

Run from the repo root: uv run scripts/prepare_asl_citizen.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets.asl_citizen import RAW_DIR, STORE_DIR, official_test_signers
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store
from isolated_sign_validation.splits import assign_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    args = parser.parse_args()

    config = PrepConfig()
    clips = assign_splits(LandmarkStore(args.store).clips.with_row_index("row"), official_test_signers(RAW_DIR))
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    prepare_store(args.store, clips.filter(pl.col("split").is_not_null()), config, out)

    data = PreparedData(out)
    print(
        data.clips.group_by("split")
        .agg(
            clips=pl.len(),
            signs=pl.col("sign").n_unique(),
            mirrored=(pl.col("dominant") == "left").mean().round(3),
            frames_median=pl.col("n_frames").median(),
            frames_p95=pl.col("n_frames").quantile(0.95),
            frames_max=pl.col("n_frames").max(),
        )
        .sort("split")
    )
    print(f"{data.frames.nbytes / 2**30:.2f} GB of frames")


if __name__ == "__main__":
    main()
