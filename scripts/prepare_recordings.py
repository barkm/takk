"""Prepare the self-recorded clips as a held-out evaluation set.

The recordings are never trained on (see ROADMAP.md), so every clip is a "test" clip. Their labels
are ASL Citizen glosses and the signs are drawn from the held-out test signs, so they are unseen
anyway. The `no_event` clips are left out: they are not clips of a sign, and every clip's sign
becomes a class of the evaluation. They stay in the landmark store, for an evaluation that can use
them as negatives.

Run from the repo root: uv run scripts/prepare_recordings.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.collection import NO_EVENT
from isolated_sign_validation.datasets import recordings
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, prepare_store
from isolated_sign_validation.splits import assign_evaluation_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=recordings.STORE_DIR)
    args = parser.parse_args()

    clips = assign_evaluation_only(LandmarkStore(args.store).clips.with_row_index("row"))
    clips = clips.filter(pl.col("sign") != NO_EVENT)
    config = PrepConfig()
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    print(f"{clips.height} clips, {clips['sign'].n_unique()} signs, {clips['signer'].n_unique()} signers -> {out}")
    prepare_store(args.store, clips, config, out)


if __name__ == "__main__":
    main()
