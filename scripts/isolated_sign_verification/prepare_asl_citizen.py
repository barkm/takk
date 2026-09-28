"""Prepare the ASL Citizen clips of the train, val and test splits for training.

Each clip gets its sign's ASL-LEX phonological features (columns `phonology.<feature>`, null for the
few signs without an ASL-LEX code), used as auxiliary training targets.

The pairs of signs that ASL-LEX links to one ASL SignBank entry (`asl_lex.signbank_twins`, one sign
form under several glosses) are written to twins.csv, so that training doesn't push them apart and
the evaluation doesn't count them as different signs. A training sign that is a twin of a held-out
sign is left out, since its clips show the held-out sign's form.

Run from the repo root: uv run scripts/isolated_sign_verification/prepare_asl_citizen.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_verification.preparation import PreparedData, add_config_arguments, config_from, prepare_store, print_summary
from isolated_sign_verification.splits import assign_splits, sign_split
from sign_data.datasets import asl_lex
from sign_data.datasets.asl_citizen import RAW_DIR, STORE_DIR, official_test_signers, read_videos
from sign_data.landmarks import LandmarkStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    add_config_arguments(parser)
    args = parser.parse_args()

    config = config_from(args)
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
    print_summary(data)


if __name__ == "__main__":
    main()
