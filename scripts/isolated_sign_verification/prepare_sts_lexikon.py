"""Prepare the Svenskt teckenspråkslexikon clips as a held-out evaluation set.

The lexicon is Swedish Sign Language and the goal vocabulary, and is never trained on, so every clip
is a "test" clip and all of its signs are unseen by construction. Sign labels get the prefix "sts:",
since a Swedish sign is a different sign from an ASL sign with the same meaning. Nothing is filtered
out: the store holds every entry, a sign class of several clips or a sign of its own (see
`datasets/sts_lexikon.py`). The clips are studio citation form.

The lexicon's models rest with their hands clasped at the waist, in frame, at about 1.0 shoulder
widths below the shoulders, right on the default `max_hand_y`. The rest is not part of the sign and
a webcam user's rest is out of frame, so the lexicon is prepared with `max_hand_y` 0.9, which hides
it in most clips.

The set serves as the glossary for self-recorded Swedish clips
(`scripts/isolated_sign_verification/evaluate_recordings.py`). Evaluating the lexicon against itself
is blocked until it has signer ids: `evaluation.py` draws every reference from a signer other than
the query's, so with one null signer for all clips no lexicon clip gets a reference.

Run from the repo root: uv run scripts/isolated_sign_verification/prepare_sts_lexikon.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_verification.preparation import PrepConfig, PreparedData, prepare_store, print_summary
from isolated_sign_verification.splits import assign_evaluation_only
from sign_data.datasets import sts_lexikon
from sign_data.landmarks import LandmarkStore


def longest_signs(data: PreparedData, n: int) -> pl.DataFrame:
    """The `n` signs whose median clip takes longest to sign, longest first.

    A lexicon entry is one sign, but a compound signed as two signs in sequence shows up here, and
    is worth viewing in the clip viewer before the numbers are trusted.
    """
    seconds = (pl.col("n_frames").median() / data.config.fps).alias("seconds")
    return data.clips.group_by("sign").agg(seconds, clips=pl.len()).sort("seconds", descending=True).head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=sts_lexikon.STORE_DIR)
    parser.add_argument("--max_hand_y", type=float, default=0.9, help="see PrepConfig and above")
    args = parser.parse_args()

    clips = assign_evaluation_only(LandmarkStore(args.store).clips.with_row_index("row"))
    clips = clips.with_columns(sign=sts_lexikon.LABEL_PREFIX + pl.col("sign"))

    config = PrepConfig(max_hand_y=args.max_hand_y)
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    print(f"{clips.height} clips of {clips['sign'].n_unique()} signs -> {out}")
    prepare_store(args.store, clips, config, out)

    data = PreparedData(out)
    print_summary(data)
    clips_per_sign = data.clips.group_by("sign").len()["len"]
    print(f"clips per sign: min {clips_per_sign.min()}, median {clips_per_sign.median()}, max {clips_per_sign.max()}")
    print(f"{(clips_per_sign > 1).sum()} signs have several clips, {(clips_per_sign == 1).sum()} a single one")
    print("the signs that take longest to sign, which may be compounds rather than one sign:")
    print(longest_signs(data, 10))


if __name__ == "__main__":
    main()
