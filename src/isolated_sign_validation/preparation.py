"""Deterministic preparation of landmark clips for training.

Hands resting low below the shoulders count as undetected. Then each clip is excluded if its hands
are missing or span implausibly little or much time. Otherwise it
is trimmed to the frames with hands (plus a margin), restored to correct proportions, short hand gaps
are interpolated, it is normalized by the shoulders, the landmark groups are selected, and it is
resampled to a common frame rate. Finally clips are mirrored so the dominant hand is always in the
right_hand slot. Missing landmarks stay NaN. Random augmentation happens at training time.
"""

import dataclasses
import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.landmarks import LANDMARK_GROUPS, LANDMARK_SLICES, MIRROR_INDEX, LandmarkStore
from isolated_sign_validation.parallel import parallel_map

_POSE = LANDMARK_SLICES["pose"].start
SHOULDERS = [_POSE + 11, _POSE + 12]  # left, right
HANDS = [LANDMARK_SLICES["left_hand"], LANDMARK_SLICES["right_hand"]]
# A clip is one-handed if both hands are detected in less than this share of its frames with a hand.
ONE_HANDED = 0.2


@dataclass(frozen=True)
class PrepConfig:
    groups: tuple[str, ...] = ("left_hand", "right_hand", "upper_body", "face_reference")
    n_coords: int = 2  # x and y; 3 adds z
    fps: float = 30.0  # clips are resampled to this frame rate
    max_frames: int = 128  # longer clips are resampled to this length
    margin: float = 0.1  # seconds kept before the first and after the last frame with a hand
    max_gap: float = 0.17  # seconds; gaps in a hand up to this long are interpolated (5 frames at 30 fps)
    min_hands: float = 0.2  # seconds; clips whose hands span less or more time are excluded
    max_hands: float = 10.0
    # shoulder widths below the shoulders; lower hands count as undetected (resting, see hide_low_hands)
    max_hand_y: float = 1.0

    def __post_init__(self):
        if not set(MIRROR_INDEX[self.landmarks]) <= set(self.landmarks):
            raise ValueError(f"groups {self.groups} must contain the mirror image of every landmark")

    @property
    def landmarks(self) -> np.ndarray:
        """Indices of the selected landmarks in the store's landmark axis."""
        return np.concatenate([LANDMARK_GROUPS[group] for group in self.groups])

    @property
    def group_slices(self) -> dict[str, slice]:
        """Where each group's landmarks are in the selected landmarks."""
        ends = np.cumsum([len(LANDMARK_GROUPS[group]) for group in self.groups])
        return {group: slice(end - len(LANDMARK_GROUPS[group]), end) for group, end in zip(self.groups, ends)}

    @property
    def mirror(self) -> np.ndarray:
        """Permutation of the selected landmarks that swaps each with its mirror image."""
        position = {index: i for i, index in enumerate(self.landmarks)}
        return np.array([position[index] for index in MIRROR_INDEX[self.landmarks]])

    def id(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:8]


def hand_presence(landmarks: np.ndarray) -> np.ndarray:
    """Whether the left and the right hand are detected in each frame, shape (n_frames, 2)."""
    return np.stack([~np.isnan(landmarks[:, hand.start, 0]) for hand in HANDS], axis=1)


def hide_low_hands(landmarks: np.ndarray, aspect: float, max_hand_y: float) -> np.ndarray:
    """Store landmarks with each hand set to undetected (NaN) in the frames where its wrist is more than
    `max_hand_y` mean shoulder widths below the mean shoulder height of the clip.

    Resting hands are out of view in close webcam framings like ASL Citizen's (whose frame bottom is
    at a median 0.7 shoulder widths below the shoulders) but detected in wider ones like MM-WLAuslan's.
    """
    xy = landmarks[..., :2] * np.array([aspect, 1.0], dtype=np.float32)
    shoulders = xy[:, SHOULDERS]
    width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    low = (xy[:, [hand.start for hand in HANDS], 1] - np.nanmean(shoulders[..., 1])) / width > max_hand_y
    out = landmarks.copy()
    for hand, hand_low in zip(HANDS, low.T):
        out[hand_low, hand] = np.nan
    return out


def fill_gaps(x: np.ndarray, present: np.ndarray, max_gap: int) -> None:
    """Linearly interpolate each hand in place over gaps of up to `max_gap` frames between detections."""
    for hand, hand_present in zip(HANDS, present.T):
        detected = np.flatnonzero(hand_present)
        for a, b in zip(detected[:-1], detected[1:]):
            if 1 < b - a <= max_gap + 1:
                w = ((np.arange(a + 1, b) - a) / (b - a))[:, None, None]
                x[a + 1 : b, hand] = (1 - w) * x[a, hand] + w * x[b, hand]


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample `x` along its first axis to `n` frames; values next to NaN become NaN."""
    if n == len(x):
        return x
    t = np.linspace(0, len(x) - 1, n)
    i = np.floor(t).astype(int)
    j = np.minimum(i + 1, len(x) - 1)
    w = (t - i).reshape(-1, *[1] * (x.ndim - 1))
    return np.where(w == 0, x[i], (1 - w) * x[i] + w * x[j]).astype(np.float32)


def prepare_clip(landmarks: np.ndarray, fps: float | None, aspect: float | None, config: PrepConfig) -> np.ndarray | None:
    """Prepare one clip of store landmarks, shape (n_frames, N_LANDMARKS, 3), without mirroring.

    `fps` and `aspect` (frame width / height) of the source video may be None if unknown; then the
    target frame rate and square frames are assumed. Returns frames of shape
    (n, len(config.landmarks), config.n_coords), or None if the clip is excluded.
    """
    fps, aspect = fps or config.fps, aspect or 1.0
    landmarks = hide_low_hands(landmarks, aspect, config.max_hand_y)
    present = hand_presence(landmarks)
    with_hands = np.flatnonzero(present.any(axis=1))
    if not len(with_hands) or not config.min_hands <= (with_hands[-1] - with_hands[0] + 1) / fps <= config.max_hands:
        return None
    margin = round(config.margin * fps)
    start, stop = max(with_hands[0] - margin, 0), min(with_hands[-1] + margin + 1, len(landmarks))
    x = np.array(landmarks[start:stop, :, : config.n_coords], dtype=np.float32)
    x[..., 0] *= aspect  # x and y are now both fractions of the frame height
    if config.n_coords == 3:
        x[..., 2] *= aspect  # MediaPipe's z has roughly the scale of x
    fill_gaps(x, present[start:stop], round(config.max_gap * fps))

    shoulders = x[:, SHOULDERS, :2]
    width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    if not width > 0:  # no pose detected
        return None
    x[..., :2] -= np.nanmean(shoulders.mean(axis=1), axis=0)
    x /= width
    x = x[:, config.landmarks]
    return resample(x, max(1, min(round(len(x) * config.fps / fps), config.max_frames)))


def mirror(frames: np.ndarray, config: PrepConfig) -> np.ndarray:
    """Mirror prepared frames (centered on the shoulders) left to right."""
    mirrored = frames[:, config.mirror].copy()
    mirrored[..., 0] *= -1
    return mirrored


def add_dominant_hands(clips: pl.DataFrame) -> pl.DataFrame:
    """Add each clip's dominant hand ("left" or "right") from its `left_frames`, `right_frames` and `both_frames`.

    A one-handed clip's dominant hand is its detected hand. A two-handed clip's is its signer's
    handedness: the hand detected in the majority of the signer's one-handed clips ("right" if none).
    """
    one_handed = pl.col("both_frames") < ONE_HANDED * (pl.col("left_frames") + pl.col("right_frames") - pl.col("both_frames"))
    detected = pl.when(pl.col("left_frames") > pl.col("right_frames")).then(pl.lit("left")).otherwise(pl.lit("right"))
    signer_hand = (
        clips.filter(one_handed)
        .group_by("signer")
        .agg(signer_hand=pl.when((detected == "left").mean() > 0.5).then(pl.lit("left")).otherwise(pl.lit("right")))
    )
    return (
        clips.join(signer_hand, on="signer", how="left", maintain_order="left")
        .with_columns(dominant=pl.when(one_handed).then(detected).otherwise(pl.col("signer_hand").fill_null("right")))
        .drop("signer_hand")
    )


@functools.cache
def _open_store(path: Path) -> LandmarkStore:
    return LandmarkStore(path)


def _prepare_rows(task: tuple[Path, list[int], PrepConfig]) -> list[tuple[np.ndarray | None, tuple[int, int, int]]]:
    """Prepared frames (or None if excluded) and hand counts (left, right, both) for store rows."""
    path, rows, config = task
    store = _open_store(path)
    results = []
    for row in rows:
        landmarks = np.asarray(store[row])
        clip = store.clips.row(row, named=True)
        aspect = clip["width"] / clip["height"] if clip["width"] else None
        present = hand_presence(hide_low_hands(landmarks, aspect or 1.0, config.max_hand_y))
        counts = (int(present[:, 0].sum()), int(present[:, 1].sum()), int(present.all(axis=1).sum()))
        results.append((prepare_clip(landmarks, clip["fps"], aspect, config), counts))
    return results


def prepare_store(store_path: Path, clips: pl.DataFrame, config: PrepConfig, out: Path, rows_per_task: int = 256) -> None:
    """Prepare `clips` (rows of the store's clip table with a `row` column of store row indices) into `out`.

    Writes frames.npy (all prepared frames back to back), clips.parquet (the included clips with
    their hand counts, dominant hand, `offset` and `n_frames` in frames.npy, and their store row as
    `store_row`) and config.json.
    """
    rows = clips["row"].to_list()
    tasks = [(store_path, rows[i : i + rows_per_task], config) for i in range(0, len(rows), rows_per_task)]
    results = [r for batch in tqdm(parallel_map(_prepare_rows, tasks), total=len(tasks), desc="preparing", unit="task") for r in batch]
    frames = [f for f, _ in results]
    counts = np.array([c for _, c in results]).reshape(-1, 3)
    clips = add_dominant_hands(
        clips.with_columns(left_frames=counts[:, 0], right_frames=counts[:, 1], both_frames=counts[:, 2])
    ).with_columns(included=pl.Series([f is not None for f in frames]))
    frames = [mirror(f, config) if dominant == "left" else f for f, dominant in zip(frames, clips["dominant"]) if f is not None]
    lengths = np.array([len(f) for f in frames])
    kept = (
        clips.filter(pl.col("included"))
        .drop("included")
        .rename({"row": "store_row"})
        .with_columns(offset=np.concatenate([[0], np.cumsum(lengths)[:-1]]), n_frames=lengths)
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", np.concatenate(frames))
    kept.write_parquet(out / "clips.parquet")
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    print(f"prepared {kept.height} of {clips.height} clips ({clips.height - kept.height} excluded) -> {out}")


class PreparedData:
    """Clips written by prepare_store; the frames are loaded into memory."""

    def __init__(self, path: Path):
        config = json.loads((path / "config.json").read_text())
        self.config = PrepConfig(**config | {"groups": tuple(config["groups"])})
        self.clips = pl.read_parquet(path / "clips.parquet")
        self.frames = np.load(path / "frames.npy")
        self._offsets = self.clips["offset"].to_numpy()
        self._n_frames = self.clips["n_frames"].to_numpy()

    def __len__(self) -> int:
        return self.clips.height

    def __getitem__(self, i: int) -> np.ndarray:
        """Frames of clip `i`, shape (n_frames, n_landmarks, n_coords), NaN where missing."""
        return self.frames[self._offsets[i] : self._offsets[i] + self._n_frames[i]]
