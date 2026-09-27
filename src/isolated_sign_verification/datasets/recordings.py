"""Adapter extracting landmarks from the self-recorded clips into a landmark store.

The recordings are collected with the web app in `collection.py`, where a signer copies reference
clips of held-out ASL Citizen signs in front of their own camera. They are never trained on: they
are an evaluation set for the setting the system is meant for.

Only the takes the signer kept are extracted; the discarded takes stay in `clips.csv`. The sign
labels are the ASL Citizen glosses, so a recording and the clips it was copied from share a label.
The signer's handedness stays in `clips.csv`, since the landmark store has no column for it.

Run from the repo root:
    uv run python -m isolated_sign_verification.datasets.recordings
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_verification.collection import RAW_DIR, read_clips
from isolated_sign_verification.extraction import extract_store

DATASET = "recordings"
STORE_DIR = Path("data/processed") / DATASET


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The takes the signer kept: columns clip_id, sign, signer and path."""
    clips = read_clips(raw_dir / "clips.csv")
    return clips.filter(pl.col("kept")).select(
        "clip_id",
        "sign",
        signer=pl.lit(f"{DATASET}:") + pl.col("signer"),
        path=pl.lit(f"{raw_dir}/videos/") + pl.col("clip_id") + ".mp4",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--store_dir", type=Path, default=STORE_DIR)
    args = parser.parse_args()

    videos = read_videos(args.raw_dir)
    # the extractor would silently turn a missing video into an empty clip
    missing = [path for path in videos["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[0]}")
    print(f"{videos.height} kept recordings by {videos['signer'].n_unique()} signers -> {args.store_dir}")
    extract_store(DATASET, videos, args.store_dir)


if __name__ == "__main__":
    main()
