"""Common, dataset-agnostic landmark format.

Every dataset is converted into a landmark store: a directory with

- ``landmarks.f32``: raw float32 array of shape (total_frames, N_LANDMARKS, 3) holding the
  x, y, z coordinates of all clips' frames back to back. Missing landmarks are NaN.
- ``clips.parquet``: one row per clip with columns ``dataset``, ``clip_id``, ``sign``,
  ``signer``, ``fps``, ``offset`` and ``n_frames``; the clip's frames are
  ``landmarks[offset : offset + n_frames]``. ``signer`` is unique across datasets.
  ``fps`` is the frame rate of the source video, null if unknown.

Landmarks are the 543 MediaPipe Holistic landmarks, in the order given by LANDMARK_SLICES.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

LANDMARK_SLICES = {
    "face": slice(0, 468),
    "left_hand": slice(468, 489),
    "pose": slice(489, 522),
    "right_hand": slice(522, 543),
}
N_LANDMARKS = 543
METADATA_COLUMNS = ["dataset", "clip_id", "sign", "signer", "fps"]

# Face mesh indices of the lips (MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_LIPS).
_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]  # fmt: skip
# Face mesh indices of the nose tip, eye corners (right outer, right inner, left inner, left outer) and chin.
_FACE_REFERENCE = [1, 33, 133, 362, 263, 152]

# Subsets of the landmarks relevant for signing, as indices into the landmark axis.
LANDMARK_GROUPS = {
    "left_hand": np.arange(LANDMARK_SLICES["left_hand"].start, LANDMARK_SLICES["left_hand"].stop),
    "right_hand": np.arange(LANDMARK_SLICES["right_hand"].start, LANDMARK_SLICES["right_hand"].stop),
    # Pose shoulders, elbows, wrists and the pose model's coarse hand points (pose indices 11-22).
    "upper_body": LANDMARK_SLICES["pose"].start + np.arange(11, 23),
    "face_reference": LANDMARK_SLICES["face"].start + np.array(_FACE_REFERENCE),
    "lips": LANDMARK_SLICES["face"].start + np.array(_LIPS),
}


def write_store(path: Path, clips: Iterable[tuple[dict, np.ndarray]]) -> None:
    """Write (metadata, landmarks) pairs to a landmark store at `path`.

    `metadata` holds METADATA_COLUMNS; `landmarks` has shape (n_frames, N_LANDMARKS, 3).
    """
    path.mkdir(parents=True, exist_ok=True)
    rows, offset = [], 0
    with open(path / "landmarks.f32", "wb") as f:
        for metadata, landmarks in clips:
            if landmarks.ndim != 3 or landmarks.shape[1:] != (N_LANDMARKS, 3):
                raise ValueError(f"clip {metadata['clip_id']}: bad landmark shape {landmarks.shape}")
            f.write(np.ascontiguousarray(landmarks, dtype=np.float32).tobytes())
            rows.append({**{c: metadata[c] for c in METADATA_COLUMNS}, "offset": offset, "n_frames": len(landmarks)})
            offset += len(landmarks)
    pl.DataFrame(rows, schema_overrides={"fps": pl.Float64}).write_parquet(path / "clips.parquet")


class LandmarkStore:
    """Read access to a landmark store; the landmarks are memory-mapped, not loaded."""

    def __init__(self, path: Path):
        self.clips = pl.read_parquet(path / "clips.parquet")
        self._offsets = self.clips["offset"].to_numpy()
        self._n_frames = self.clips["n_frames"].to_numpy()
        shape = (int(self._n_frames.sum()), N_LANDMARKS, 3)
        self.landmarks = np.memmap(path / "landmarks.f32", dtype=np.float32, mode="r", shape=shape)

    def __len__(self) -> int:
        return self.clips.height

    def __getitem__(self, i: int) -> np.ndarray:
        """Landmarks of clip `i` (row `i` of `clips`), shape (n_frames, N_LANDMARKS, 3)."""
        return self.landmarks[self._offsets[i] : self._offsets[i] + self._n_frames[i]]
