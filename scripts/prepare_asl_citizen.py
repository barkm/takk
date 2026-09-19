"""Prepare the ASL Citizen clips of the train, val and test splits for training.

Each clip gets its sign's ASL-LEX phonological features (columns `phonology.<feature>`, null for the
few signs without an ASL-LEX code), used as auxiliary training targets.

The pairs of signs that ASL-LEX links to one ASL SignBank entry (`asl_lex.signbank_twins`, one sign
form under several glosses) are written to twins.csv, so that training doesn't push them apart and
the evaluation doesn't count them as different signs. A training sign that is a twin of a held-out
sign is left out, since its clips show the held-out sign's form (see ROADMAP.md).

Run from the repo root: uv run scripts/prepare_asl_citizen.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import asl_lex
from isolated_sign_validation.datasets.asl_citizen import RAW_DIR, STORE_DIR, official_test_signers, read_videos
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store
from isolated_sign_validation.splits import assign_splits, sign_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    args = parser.parse_args()

    config = PrepConfig()
    clips = assign_splits(LandmarkStore(args.store).clips.with_row_index("row"), official_test_signers(RAW_DIR))
    codes = read_videos(RAW_DIR).select(sign="Gloss", Code="ASL-LEX Code").unique()
    phonology = codes.join(asl_lex.read_phonology(asl_lex.RAW_DIR), on="Code", how="left").drop("Code")
    clips = clips.join(phonology, on="sign", how="left", maintain_order="left")
    twins = asl_lex.signbank_twins(asl_lex.RAW_DIR, codes)
    near_held_out = {a for pair in twins for a in pair if sign_split(a) == "train" and any(sign_split(b) != "train" for b in pair)}
    print(f"{len(twins)} twin pairs linked to one SignBank entry; {len(near_held_out)} training signs are twins of held-out signs and left out")
    clips = clips.filter(pl.col("split").is_not_null() & ~pl.col("sign").is_in(list(near_held_out)))
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    prepare_store(args.store, clips, config, out)
    pl.DataFrame(sorted(twins), schema=["sign_a", "sign_b"], orient="row").write_csv(out / "twins.csv")

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
