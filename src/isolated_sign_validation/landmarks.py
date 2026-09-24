"""Common, dataset-agnostic landmark format.

Every dataset is converted into a landmark store: a directory with

- ``landmarks.f32``: raw float32 array of shape (total_frames, N_LANDMARKS, 3) holding the
  x, y, z coordinates of all clips' frames back to back. Missing landmarks are NaN.
- ``clips.parquet``: one row per clip with columns ``dataset``, ``clip_id``, ``sign``,
  ``signer``, ``fps``, ``width``, ``height``, ``offset`` and ``n_frames``; the clip's frames are
  ``landmarks[offset : offset + n_frames]``. ``signer`` is unique across datasets, or null if
  the dataset has no signer ids.
  ``fps`` is the frame rate and ``width`` and ``height`` the frame size of the source video,
  null if unknown. Landmark x and y are fractions of the frame width and height.

Landmarks are the 543 MediaPipe Holistic landmarks, in the order given by LANDMARK_SLICES.
"""

import itertools
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

LANDMARK_SLICES = {
    "face": slice(0, 468),
    "left_hand": slice(468, 489),
    "pose": slice(489, 522),
    "right_hand": slice(522, 543),
}
N_LANDMARKS = 543
METADATA_COLUMNS = ["dataset", "clip_id", "sign", "signer", "fps", "width", "height"]


class VideoInfo(NamedTuple):
    """The frame rate and frame size a clip's landmarks were extracted from. It lives here rather
    than in `extraction.py` so that reading landmarks does not import MediaPipe: the practice app
    takes them from a browser and never extracts any (see step 18 of ROADMAP-takk.md)."""

    fps: float
    width: int  # of the decoded frames, which the landmark coordinates are fractions of
    height: int

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

# MediaPipe HandLandmarksConnections.HAND_CONNECTIONS
_HAND_EDGES = [(0, 1), (0, 17), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (5, 9), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (13, 17), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]  # fmt: skip
# MediaPipe PoseLandmarksConnections.POSE_LANDMARKS between shoulders, arms and pose hand points
_UPPER_BODY_EDGES = [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20)]  # fmt: skip
# MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_LIPS
_LIPS_EDGES = [(0, 267), (13, 312), (14, 317), (17, 314), (37, 0), (39, 37), (40, 39), (61, 146), (61, 185), (78, 95), (78, 191), (80, 81), (81, 82), (82, 13), (84, 17), (87, 14), (88, 178), (91, 181), (95, 88), (146, 91), (178, 87), (181, 84), (185, 40), (191, 80), (267, 269), (269, 270), (270, 409), (310, 415), (311, 310), (312, 311), (314, 405), (317, 402), (318, 324), (321, 375), (324, 308), (375, 291), (402, 318), (405, 321), (409, 291), (415, 308)]  # fmt: skip

# Mirror-image pairs: the same point on the other side of the body. Pose indices: shoulders, elbows,
# wrists, pinkies, index fingers, thumbs. Face mesh indices: eye corners, then the lip contours
# (outer lower, outer upper, inner lower, inner upper), each from the mouth corner to the midline.
_POSE_MIRROR_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
_FACE_MIRROR_PAIRS = [
    (33, 263), (133, 362),
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
    (185, 409), (40, 270), (39, 269), (37, 267),
    (78, 308), (95, 324), (88, 318), (178, 402), (87, 317),
    (191, 415), (80, 310), (81, 311), (82, 312),
]  # fmt: skip


def _mirror_index() -> np.ndarray:
    index = np.arange(N_LANDMARKS)  # points on the midline are their own mirror image
    left, right = LANDMARK_SLICES["left_hand"], LANDMARK_SLICES["right_hand"]
    index[left], index[right] = np.arange(right.start, right.stop), np.arange(left.start, left.stop)
    for offset, pairs in [(LANDMARK_SLICES["pose"].start, _POSE_MIRROR_PAIRS), (LANDMARK_SLICES["face"].start, _FACE_MIRROR_PAIRS)]:
        for a, b in pairs:
            index[offset + a], index[offset + b] = offset + b, offset + a
    return index


# For each landmark, the index of its mirror image. Defined for the landmarks in LANDMARK_GROUPS.
MIRROR_INDEX = _mirror_index()

# Connections between the landmarks of a group, as pairs of indices into the landmark axis.
SKELETON_EDGES = {
    "left_hand": np.array(_HAND_EDGES) + LANDMARK_SLICES["left_hand"].start,
    "right_hand": np.array(_HAND_EDGES) + LANDMARK_SLICES["right_hand"].start,
    "upper_body": np.array(_UPPER_BODY_EDGES) + LANDMARK_SLICES["pose"].start,
    "lips": np.array(_LIPS_EDGES) + LANDMARK_SLICES["face"].start,
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
    schema = {"signer": pl.String, "fps": pl.Float64, "width": pl.Int64, "height": pl.Int64}  # so all-null columns keep their type
    pl.DataFrame(rows, schema_overrides=schema).write_parquet(path / "clips.parquet")


def merge_stores(paths: Sequence[Path], out: Path) -> None:
    """Concatenate the landmark stores at `paths`, in order, into one store at `out`."""
    out.mkdir(parents=True, exist_ok=True)
    parts, offset = [], 0
    with open(out / "landmarks.f32", "wb") as f:
        for path in paths:
            clips = pl.read_parquet(path / "clips.parquet")
            with open(path / "landmarks.f32", "rb") as part:
                shutil.copyfileobj(part, f, length=64 * 2**20)
            parts.append(clips.with_columns(pl.col("offset") + offset))
            offset += int(clips["n_frames"].sum())
    pl.concat(parts).write_parquet(out / "clips.parquet")


def write_store_resumable[T](
    path: Path,
    items: Sequence[T],
    convert: Callable[[Sequence[T]], Iterable[tuple[dict, np.ndarray]]],
    chunk_size: int = 1024,
) -> None:
    """Write a landmark store at `path` in chunks, so that an interrupted run can be resumed.

    `convert` turns items into (metadata, landmarks) pairs, one per item and in order. Every
    `chunk_size` items are written as a separate store in ``path/chunks/``; when run again with
    the same `items`, finished chunks are skipped. At the end the chunks are merged into the
    store and removed.
    """
    if (path / "clips.parquet").exists():
        raise FileExistsError(f"{path} already holds a finished store; delete it to rebuild")
    chunks = [(path / "chunks" / f"{i:05d}", items[start : start + chunk_size]) for i, start in enumerate(range(0, len(items), chunk_size))]
    # A chunk is finished once its clips.parquet exists: write_store writes it last.
    pending = [(chunk, chunk_items) for chunk, chunk_items in chunks if not (chunk / "clips.parquet").exists()]
    print(f"{len(chunks) - len(pending)} of {len(chunks)} chunks already written")
    clips = iter(convert([item for _, chunk_items in pending for item in chunk_items]))
    for chunk, chunk_items in pending:
        write_store(chunk, itertools.islice(clips, len(chunk_items)))
    merge_stores([chunk for chunk, _ in chunks], path)
    shutil.rmtree(path / "chunks")


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
