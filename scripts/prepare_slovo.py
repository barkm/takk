"""Prepare the Slovo clips as a held-out cross-language evaluation set.

Slovo is Russian Sign Language and is never trained on (see ROADMAP.md), so every clip is a "test"
clip and all of its signs are unseen by construction. Classes whose label is a phrase rather than a
single sign are left out, since the task is validating one sign: labels without a single-word gloss
(`slovo.is_single_sign`), and classes whose clips take much longer to sign than usual
(`--max_median_seconds`), which catches phrases that the label doesn't reveal. Sign labels get the
prefix "rsl:", since a Russian sign is a different sign from an ASL sign with the same meaning.

Run from the repo root: uv run scripts/prepare_slovo.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import slovo
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store
from isolated_sign_validation.splits import assign_evaluation_only

LABEL_PREFIX = "rsl:"


def long_signs(data: PreparedData, max_median_seconds: float) -> pl.DataFrame:
    """The signs whose median clip takes longer than `max_median_seconds` to sign, longest first."""
    seconds = (pl.col("n_frames").median() / data.config.fps).alias("seconds")
    return data.clips.group_by("sign").agg(seconds).filter(pl.col("seconds") > max_median_seconds).sort("seconds", descending=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=slovo.STORE_DIR)
    parser.add_argument(
        "--max_median_seconds",
        type=float,
        default=3.0,
        help="leave out classes whose median clip takes longer than this to sign (0: keep all)",
    )
    args = parser.parse_args()

    clips = assign_evaluation_only(LandmarkStore(args.store).clips.with_row_index("row"))
    clips = clips.with_columns(sign=LABEL_PREFIX + pl.col("sign"))
    phrases = {sign for sign in clips["sign"].unique() if not slovo.is_single_sign(sign.removeprefix(LABEL_PREFIX))}
    print(f"{len(phrases)} of {clips['sign'].n_unique()} classes have a phrase label and are left out")
    clips = clips.filter(~pl.col("sign").is_in(list(phrases)))

    config = PrepConfig()
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    prepare_store(args.store, clips, config, out)

    if args.max_median_seconds:
        long = long_signs(PreparedData(out), args.max_median_seconds)
        print(f"{long.height} classes take a median of over {args.max_median_seconds} s to sign and are left out:")
        print(long)
        if long.height:  # prepare again without them, so their frames are not kept either
            prepare_store(args.store, clips.filter(~pl.col("sign").is_in(long["sign"])), config, out)

    data = PreparedData(out)
    print(
        data.clips.select(
            clips=pl.len(),
            signs=pl.col("sign").n_unique(),
            signers=pl.col("signer").n_unique(),
            mirrored=(pl.col("dominant") == "left").mean().round(3),
            frames_median=pl.col("n_frames").median(),
            frames_p95=pl.col("n_frames").quantile(0.95),
            frames_max=pl.col("n_frames").max(),
        )
    )
    signers_per_sign = data.clips.group_by("sign").agg(signers=pl.col("signer").n_unique())["signers"]
    print(f"signers per sign: min {signers_per_sign.min()}, median {signers_per_sign.median()}")
    print(f"{data.frames.nbytes / 2**30:.2f} GB of frames")


if __name__ == "__main__":
    main()
