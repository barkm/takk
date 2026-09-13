"""Prepare the MM-WLAuslan clips for training alongside ASL Citizen.

MM-WLAuslan is used for training only (see ROADMAP.md): all clips are "train", except those of signs
whose gloss or English keyword matches an ASL Citizen val or test sign by label, which could look like
held-out signs; they are prepared too, with a null split. Sign labels get the prefix "auslan:", since
an Auslan sign is a different sign from an ASL sign with the same English label.

Run from the repo root: uv run scripts/prepare_mm_wlauslan.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import asl_citizen, mm_wlauslan
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store
from isolated_sign_validation.splits import assign_training_only, matching_signs, sign_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=mm_wlauslan.STORE_DIR)
    args = parser.parse_args()

    held_out = [sign for sign in LandmarkStore(asl_citizen.STORE_DIR).clips["sign"].unique() if sign_split(sign) != "train"]
    excluded = matching_signs(mm_wlauslan.sign_words(mm_wlauslan.RAW_DIR), held_out)
    clips = assign_training_only(LandmarkStore(args.store).clips.with_row_index("row"), excluded)
    print(f"{clips.filter(pl.col('split').is_null())['sign'].n_unique()} of {clips['sign'].n_unique()} signs match one of "
          f"{len(held_out)} ASL Citizen val/test signs and are left out of training")  # fmt: skip

    config = PrepConfig()
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    prepare_store(args.store, clips.with_columns(sign="auslan:" + pl.col("sign")), config, out)

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
