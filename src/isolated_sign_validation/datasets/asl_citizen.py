"""Adapter extracting landmarks from the ASL Citizen videos into a landmark store.

Extraction takes about half a day; an interrupted run resumes where it left off when run again.
Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.asl_citizen             # all videos
    uv run python -m isolated_sign_validation.datasets.asl_citizen --signs 10  # all videos of 10 random signs
"""

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.extraction import download_model, extract_landmarks, silence_native_logs
from isolated_sign_validation.landmarks import write_store_resumable
from isolated_sign_validation.parallel import parallel_map

DATASET = "asl_citizen"
RAW_DIR = Path("data/raw/asl-citizen/ASL_Citizen")
STORE_DIR = Path("data/processed") / DATASET
# Holistic extraction uses ~1.7 cores per process; more workers than this are slower (see ROADMAP.md).
MAX_WORKERS = 8


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """All videos listed in the official train, val and test splits."""
    return pl.concat([pl.read_csv(raw_dir / "splits" / f"{split}.csv") for split in ("train", "val", "test")])


def official_test_signers(raw_dir: Path) -> set[str]:
    """Signers of the official test split, as signer ids of the landmark store."""
    test = pl.read_csv(raw_dir / "splits" / "test.csv")
    return {f"{DATASET}:{participant}" for participant in test["Participant ID"].unique()}


def extract_clips(videos: Sequence[dict], raw_dir: Path) -> Iterator[tuple[dict, np.ndarray]]:
    """Extract (metadata, landmarks) for `videos` (rows of the split CSVs), in order."""
    paths = [raw_dir / "videos" / video["Video file"] for video in videos]
    results = parallel_map(extract_landmarks, paths, max_workers=MAX_WORKERS, initializer=silence_native_logs)
    for video, (landmarks, info) in zip(videos, results):
        metadata = {
            "dataset": DATASET,
            "clip_id": Path(video["Video file"]).stem,
            "sign": video["Gloss"],
            "signer": f"{DATASET}:{video['Participant ID']}",
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        }
        yield metadata, landmarks


def convert(videos: pl.DataFrame, raw_dir: Path, store_dir: Path) -> None:
    """Extract landmarks for `videos` into a landmark store, resuming an interrupted run."""
    download_model()
    write_store_resumable(
        store_dir,
        videos.to_dicts(),
        lambda batch: tqdm(extract_clips(batch, raw_dir), total=len(batch), desc=DATASET, unit="clip"),
    )


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
    convert(videos, RAW_DIR, store_dir)


if __name__ == "__main__":
    main()
