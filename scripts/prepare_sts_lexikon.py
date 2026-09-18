"""Prepare the Svenskt teckenspråkslexikon clips as a held-out evaluation set.

The lexicon is Swedish Sign Language and the goal vocabulary, and is never trained on (see
ROADMAP.md), so every clip is a "test" clip and all of its signs are unseen by construction. Sign
labels get the prefix "sts:", since a Swedish sign is a different sign from an ASL sign with the
same meaning. The store already holds only the clips of classes with at least two recordings, so
nothing is filtered out here; the clips are studio citation form, which makes this an optimistic
bound relative to webcam signing.

The clips have no signer ids yet (see ROADMAP.md), so a k-shot evaluation on this set cannot
require the reference clips to come from another signer, and a class recorded twice by the same
model gives an easier trial than the protocol intends.

Run from the repo root: uv run scripts/prepare_sts_lexikon.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import sts_lexikon
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store
from isolated_sign_validation.splits import assign_evaluation_only

LABEL_PREFIX = "sts:"


def longest_signs(data: PreparedData, n: int) -> pl.DataFrame:
    """The `n` signs whose median clip takes longest to sign, longest first.

    A lexicon entry is one sign, but a compound signed as two signs in sequence shows up here, and
    is worth viewing in the clip viewer before the numbers are trusted (see ROADMAP.md).
    """
    seconds = (pl.col("n_frames").median() / data.config.fps).alias("seconds")
    return data.clips.group_by("sign").agg(seconds, clips=pl.len()).sort("seconds", descending=True).head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=sts_lexikon.STORE_DIR)
    args = parser.parse_args()

    clips = assign_evaluation_only(LandmarkStore(args.store).clips.with_row_index("row"))
    clips = clips.with_columns(sign=LABEL_PREFIX + pl.col("sign"))

    config = PrepConfig()
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    print(f"{clips.height} clips of {clips['sign'].n_unique()} signs -> {out}")
    prepare_store(args.store, clips, config, out)

    data = PreparedData(out)
    print(
        data.clips.select(
            clips=pl.len(),
            signs=pl.col("sign").n_unique(),
            mirrored=(pl.col("dominant") == "left").mean().round(3),
            frames_median=pl.col("n_frames").median(),
            frames_p95=pl.col("n_frames").quantile(0.95),
            frames_max=pl.col("n_frames").max(),
        )
    )
    clips_per_sign = data.clips.group_by("sign").len()["len"]
    print(f"clips per sign: min {clips_per_sign.min()}, median {clips_per_sign.median()}, max {clips_per_sign.max()}")
    print("the signs that take longest to sign, which may be compounds rather than one sign:")
    print(longest_signs(data, 10))
    print(f"{data.frames.nbytes / 2**30:.2f} GB of frames")


if __name__ == "__main__":
    main()
