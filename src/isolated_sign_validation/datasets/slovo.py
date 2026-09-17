"""Adapter extracting landmarks from the Slovo videos into a landmark store.

Slovo is Russian Sign Language and is used as a held-out cross-language evaluation set (see
ROADMAP.md), so its own train/test split is ignored and only the signer ids matter. The 400
`no_event` clips (no signing) are not clips of a sign and are left out. Expects the downloaded zip
unzipped in place, e.g. data/raw/slovo/train/<attachment id>.mp4.

Extraction takes about 2 hours; an interrupted run resumes where it left off when run again.
Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.slovo             # all videos
    uv run python -m isolated_sign_validation.datasets.slovo --signs 10  # all videos of 10 random signs
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

DATASET = "slovo"
RAW_DIR = Path("data/raw/slovo")
STORE_DIR = Path("data/processed") / DATASET
# Holistic extraction uses ~1.7 cores per process; more workers than this are slower (see ROADMAP.md).
MAX_WORKERS = 8
NO_EVENT = "no_event"  # the class of clips without signing


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The annotated videos, without the `no_event` clips: columns clip_id, sign, signer and path."""
    annotations = pl.read_csv(raw_dir / "annotations.csv", separator="\t")
    folder = pl.when(pl.col("train")).then(pl.lit("train")).otherwise(pl.lit("test"))
    return annotations.filter(pl.col("text") != NO_EVENT).select(
        clip_id=pl.col("attachment_id"),
        sign=pl.col("text"),
        signer=pl.lit(f"{DATASET}:") + pl.col("user_id"),
        path=pl.lit(f"{raw_dir}/") + folder + "/" + pl.col("attachment_id") + ".mp4",
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
    parser.add_argument(
        "--signs", type=int, help="only extract all videos of this many randomly chosen signs, into a separate store"
    )
    args = parser.parse_args()

    videos, store_dir = read_videos(RAW_DIR), STORE_DIR
    if args.signs:
        signs = videos["sign"].unique().sort().sample(args.signs, seed=0).sort().to_list()
        videos = videos.filter(pl.col("sign").is_in(signs))
        store_dir = STORE_DIR.with_name(f"{DATASET}_{args.signs}_signs")
    # the extractor would silently turn a missing video into an empty clip
    missing = [path for path in videos["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[0]}; download and unzip them first")
    print(f"{videos.height} videos -> {store_dir}")
    convert(videos, store_dir)


if __name__ == "__main__":
    main()
