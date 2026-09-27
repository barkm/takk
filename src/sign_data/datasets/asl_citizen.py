"""Adapter extracting landmarks from the ASL Citizen videos into a landmark store.

Extraction takes about half a day; an interrupted run resumes where it left off when run again.
Run from the repo root:
    uv run python -m sign_data.datasets.asl_citizen             # all videos
    uv run python -m sign_data.datasets.asl_citizen --signs 10  # all videos of 10 random signs
"""

import argparse
from pathlib import Path

import polars as pl

from sign_data.extraction import extract_store

DATASET = "asl_citizen"
RAW_DIR = Path("data/raw/asl-citizen/ASL_Citizen")
STORE_DIR = Path("data/processed") / DATASET


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """All videos listed in the official train, val and test splits: their columns, plus clip_id,
    sign, signer and path as the other adapters give them."""
    videos = pl.concat([pl.read_csv(raw_dir / "splits" / f"{split}.csv") for split in ("train", "val", "test")])
    return videos.with_columns(
        clip_id=pl.col("Video file").str.strip_suffix(".mp4"),
        sign=pl.col("Gloss"),
        signer=pl.lit(f"{DATASET}:") + pl.col("Participant ID").cast(pl.String),
        path=pl.lit(f"{raw_dir}/videos/") + pl.col("Video file"),
    )


def official_test_signers(raw_dir: Path) -> set[str]:
    """Signers of the official test split, as signer ids of the landmark store."""
    test = pl.read_csv(raw_dir / "splits" / "test.csv")
    return {f"{DATASET}:{participant}" for participant in test["Participant ID"].unique()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--signs", type=int, help="only extract all videos of this many randomly chosen signs, into a separate store"
    )
    args = parser.parse_args()

    videos, store_dir = read_videos(RAW_DIR), STORE_DIR
    if args.signs:
        signs = videos["Gloss"].unique().sort().sample(args.signs, seed=0).sort().to_list()
        videos = videos.filter(pl.col("Gloss").is_in(signs))
        store_dir = STORE_DIR.with_name(f"{DATASET}_{args.signs}_signs")
        print(f"{videos.height} videos of the signs {', '.join(signs)} -> {store_dir}")
    extract_store(DATASET, videos, store_dir)


if __name__ == "__main__":
    main()
