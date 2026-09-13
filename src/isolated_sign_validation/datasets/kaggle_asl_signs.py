"""Adapter converting the Kaggle ASL Signs dataset into a landmark store.

Run from the repo root: uv run python -m isolated_sign_validation.datasets.kaggle_asl_signs
"""

import os
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS, write_store
from isolated_sign_validation.parallel import parallel_map

DATASET = "kaggle_asl_signs"
RAW_DIR = Path("data/raw/asl-signs")
STORE_DIR = Path("data/processed") / DATASET


def read_clip(path: Path) -> np.ndarray:
    """Read one Kaggle parquet file into an array of shape (n_frames, N_LANDMARKS, 3).

    Frames are kept in order; gaps in the frame numbers are closed.
    """
    df = pl.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    type_offsets = {type_: s.start for type_, s in LANDMARK_SLICES.items()}
    landmark = df["type"].replace_strict(type_offsets, return_dtype=pl.Int64) + df["landmark_index"]
    frame = df["frame"].rank("dense").cast(pl.Int64) - 1
    landmarks = np.full((df["frame"].n_unique(), N_LANDMARKS, 3), np.nan, dtype=np.float32)
    landmarks[frame.to_numpy(), landmark.to_numpy()] = df.select("x", "y", "z").to_numpy()
    return landmarks


def convert(train: pl.DataFrame, raw_dir: Path, store_dir: Path) -> None:
    """Convert the clips listed in `train` (rows of Kaggle's train.csv) into a landmark store."""
    metadata = (
        {
            "dataset": DATASET,
            "clip_id": str(row["sequence_id"]),
            "sign": row["sign"],
            "signer": f"{DATASET}:{row['participant_id']}",
            # not published for this dataset
            "fps": None,
            "width": None,
            "height": None,
        }
        for row in train.iter_rows(named=True)
    )
    os.environ["POLARS_MAX_THREADS"] = "1"  # for the workers: parallelism comes from processes
    clips = zip(metadata, parallel_map(read_clip, [raw_dir / p for p in train["path"]], chunksize=16))
    write_store(store_dir, tqdm(clips, total=train.height, desc=DATASET, unit="clip"))


if __name__ == "__main__":
    convert(pl.read_csv(RAW_DIR / "train.csv"), RAW_DIR, STORE_DIR)
