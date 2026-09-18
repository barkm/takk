"""Prepare the Svenskt teckenspråkslexikon clips as a held-out evaluation set.

The lexicon is Swedish Sign Language and the goal vocabulary, and is never trained on (see
ROADMAP.md), so every clip is a "test" clip and all of its signs are unseen by construction. Sign
labels get the prefix "sts:", since a Swedish sign is a different sign from an ASL sign with the
same meaning. Nothing is filtered out: the store holds every entry, a sign class of several clips
or a sign of its own (see `datasets/sts_lexikon.py`). The clips are studio citation form.

The set serves as the glossary for self-recorded Swedish clips (`scripts/evaluate_recordings.py`).
Evaluating the lexicon against itself is blocked until it has signer ids (see ROADMAP.md):
`evaluation.py` draws every reference from a signer other than the query's, so with one null signer
for all clips no lexicon clip gets a reference.

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
    clips = clips.with_columns(sign=sts_lexikon.LABEL_PREFIX + pl.col("sign"))

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
    print(f"{(clips_per_sign > 1).sum()} signs have several clips, {(clips_per_sign == 1).sum()} a single one")
    print("the signs that take longest to sign, which may be compounds rather than one sign:")
    print(longest_signs(data, 10))
    print(f"{data.frames.nbytes / 2**30:.2f} GB of frames")


if __name__ == "__main__":
    main()
