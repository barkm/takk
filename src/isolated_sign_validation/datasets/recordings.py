"""Adapter extracting landmarks from the self-recorded clips into a landmark store.

The recordings are collected with the web app in `collection.py`, where a signer copies reference
clips of held-out ASL Citizen signs in front of their own camera. They are never trained on: they
are an evaluation set for the setting the system is meant for (see ROADMAP.md).

Only the takes the signer kept are extracted; the discarded takes stay in `clips.csv`. The sign
labels are the ASL Citizen glosses, so a recording and the clips it was copied from share a label.
The signer's handedness stays in `clips.csv`, since the landmark store has no column for it.

Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.recordings
"""

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.collection import RAW_DIR, SCHEMA
from isolated_sign_validation.extraction import download_model, extract_landmarks, silence_native_logs
from isolated_sign_validation.landmarks import write_store_resumable
from isolated_sign_validation.parallel import parallel_map

DATASET = "recordings"
STORE_DIR = Path("data/processed") / DATASET
# Holistic extraction uses ~1.7 cores per process; more workers than this are slower (see ROADMAP.md).
MAX_WORKERS = 8


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The takes the signer kept: columns clip_id, sign, signer and path."""
    clips = pl.read_csv(raw_dir / "clips.csv", schema=SCHEMA)
    return clips.filter(pl.col("kept")).select(
        "clip_id",
        "sign",
        signer=pl.lit(f"{DATASET}:") + pl.col("signer"),
        path=pl.lit(f"{raw_dir}/videos/") + pl.col("clip_id") + ".mp4",
    )


def extract_clips(videos: Sequence[dict]) -> Iterator[tuple[dict, np.ndarray]]:
    """Extract (metadata, landmarks) for `videos` (rows of read_videos), in order."""
    results = parallel_map(extract_landmarks, [Path(video["path"]) for video in videos], max_workers=MAX_WORKERS, initializer=silence_native_logs)  # fmt: skip
    for video, (landmarks, info) in zip(videos, results):
        metadata = {
            "dataset": DATASET,
            "clip_id": video["clip_id"],
            "sign": video["sign"],
            "signer": video["signer"],
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        }
        yield metadata, landmarks


def convert(videos: pl.DataFrame, store_dir: Path) -> None:
    """Extract landmarks for `videos` into a landmark store, resuming an interrupted run."""
    download_model()
    write_store_resumable(
        store_dir,
        videos.to_dicts(),
        lambda batch: tqdm(extract_clips(batch), total=len(batch), desc=DATASET, unit="clip"),
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
    convert(videos, args.store_dir)


if __name__ == "__main__":
    main()
