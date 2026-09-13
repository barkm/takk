"""PyTorch dataset of prepared clips, with random augmentation for training.

Items are frames with missing landmarks set to 0, plus per-frame flags for which hands are present.
`collate` pads a batch to its longest clip and adds a mask of the real frames.
"""

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import torch

from isolated_sign_validation.preparation import PreparedData, resample


@dataclass(frozen=True)
class AugmentConfig:
    rotation: float = 20.0  # degrees, up to this much in either direction
    scale: float = 0.3  # relative, up to this much larger or smaller
    shift: float = 0.15  # in shoulder widths, in each direction
    shear: float = 0.2
    speed: tuple[float, float] = (0.7, 1.4)  # playback speed range
    frame_drop: float = 0.2  # probability of dropping each frame
    hand_drop: float = 0.25  # probability of blanking each hand for a random stretch of frames


def affine(frames: np.ndarray, matrix: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Transform x, y of all landmarks: x, y ← matrix @ (x, y) + shift."""
    out = frames.copy()
    out[..., :2] = frames[..., :2] @ matrix.T + shift
    return out


def drop_frames(frames: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Keep the frames where `keep` is true, but always at least one."""
    return frames[keep] if keep.any() else frames[:1]


def drop_hand(frames: np.ndarray, hand: slice, start: int, stop: int) -> np.ndarray:
    """Blank a hand (set it to NaN, like an undetected hand) in frames `start` to `stop`."""
    out = frames.copy()
    out[start:stop, hand] = np.nan
    return out


def augment(frames: np.ndarray, rng: np.random.Generator, config: AugmentConfig, hands: list[slice], max_frames: int) -> np.ndarray:
    """Randomly augment prepared frames (NaN where missing)."""
    angle = np.radians(rng.uniform(-config.rotation, config.rotation))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shear = np.array([[1, rng.uniform(-config.shear, config.shear)], [0, 1]])
    scale = rng.uniform(1 - config.scale, 1 + config.scale)
    frames = affine(frames, scale * rotation @ shear, rng.uniform(-config.shift, config.shift, size=2))

    n = round(len(frames) / rng.uniform(*config.speed))
    frames = resample(frames, max(1, min(n, max_frames)))
    frames = drop_frames(frames, rng.random(len(frames)) >= config.frame_drop)
    for hand in hands:
        if rng.random() < config.hand_drop:
            length = rng.integers(1, max(2, len(frames) // 2))
            start = rng.integers(0, len(frames) - length + 1)
            frames = drop_hand(frames, hand, start, start + length)
    return frames


class SignDataset(torch.utils.data.Dataset):
    """The prepared clips of one split, labeled by sign (indices into `signs`); optionally only of
    `only_signs`, and without the clips of `exclude_signers`."""

    def __init__(
        self,
        data: PreparedData,
        split: str,
        augment: AugmentConfig | None = None,
        exclude_signers: Collection[str] = (),
        only_signs: Collection[str] | None = None,
    ):
        self.data = data
        in_split = (data.clips["split"] == split).fill_null(False)
        selected = in_split & ~data.clips["signer"].is_in(list(exclude_signers)).fill_null(False)
        if only_signs is not None:
            selected &= data.clips["sign"].is_in(list(only_signs))
        self.positions = np.flatnonzero(selected.to_numpy())
        clip_signs = data.clips["sign"].to_numpy()[self.positions]
        self.signs = sorted(set(clip_signs))
        self.labels = np.searchsorted(self.signs, clip_signs)
        self.augment = augment
        self.hands = [data.config.group_slices[hand] for hand in ("left_hand", "right_hand")]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> dict:
        frames = self.data[self.positions[i]]
        if self.augment:
            # torch's generator is seeded differently in each DataLoader worker and epoch
            rng = np.random.default_rng(torch.randint(2**62, ()).item())
            frames = augment(frames, rng, self.augment, self.hands, self.data.config.max_frames)
        hands = np.stack([~np.isnan(frames[:, hand.start, 0]) for hand in self.hands], axis=1)
        return {
            "frames": torch.from_numpy(np.nan_to_num(frames)),
            "hands": torch.from_numpy(hands),
            "label": int(self.labels[i]),
        }


def collate(items: list[dict]) -> dict:
    """Pad items to the longest clip: frames (B, T, L, C), hands (B, T, 2), mask (B, T), labels (B,)."""
    lengths = [len(item["frames"]) for item in items]
    frames = torch.zeros(len(items), max(lengths), *items[0]["frames"].shape[1:])
    hands = torch.zeros(len(items), max(lengths), 2, dtype=torch.bool)
    mask = torch.zeros(len(items), max(lengths), dtype=torch.bool)
    for b, (item, n) in enumerate(zip(items, lengths)):
        frames[b, :n], hands[b, :n], mask[b, :n] = item["frames"], item["hands"], True
    return {"frames": frames, "hands": hands, "mask": mask, "labels": torch.tensor([item["label"] for item in items])}
