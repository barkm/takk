"""Baselines without training: a hand-crafted embedding and dynamic time warping (DTW).

Both work on prepared clips (normalized, dominant hand in the right_hand slot) and produce a
similarity matrix for the evaluation harness.
"""

import warnings

import numpy as np
import torch

from isolated_sign_validation.preparation import PrepConfig, resample

# Pose wrists within the upper_body group (pose indices 11-22): left wrist 15, right wrist 16
_WRISTS = {"left_hand": 4, "right_hand": 5}


def hand_embedding(frames: np.ndarray, config: PrepConfig) -> np.ndarray:
    """The dominant hand's mean shape relative to its wrist, its mean position and its wrist path at 8 time points."""
    hand = frames[:, config.group_slices["right_hand"]]
    with warnings.catch_warnings():  # a hand that is never detected gives all-NaN means, set to 0 below
        warnings.simplefilter("ignore", RuntimeWarning)
        shape = np.nanmean(hand - hand[:, :1], axis=0).ravel()
        position = np.nanmean(hand[:, 0], axis=0)
    return np.nan_to_num(np.concatenate([shape, position, resample(hand[:, 0], 8).ravel()]))


def dtw_features(frames: np.ndarray, config: PrepConfig, n_frames: int = 32) -> np.ndarray:
    """Per-frame features for DTW, shape (n_frames, 108): both hands (x, y) and the upper body.

    In frames where a hand is missing, all its points are put at the pose wrist of its side.
    """
    upper_body = frames[:, config.group_slices["upper_body"]]
    hands = []
    for group, wrist in _WRISTS.items():
        hand = frames[:, config.group_slices[group]].copy()
        missing = np.isnan(hand[:, 0, 0])
        hand[missing] = upper_body[missing, wrist][:, None]
        hands.append(hand)
    features = np.concatenate([*hands, upper_body], axis=1).reshape(len(frames), -1)
    return np.nan_to_num(resample(features, n_frames))


def dtw_from_cost(cost: torch.Tensor) -> torch.Tensor:
    """DTW distances from a batch of cost matrices (B, n, m): the cheapest monotonic alignment path's cost / (n + m)."""
    b, n, m = cost.shape
    total = torch.full((b, n + 1, m + 1), torch.inf, dtype=cost.dtype, device=cost.device)
    total[:, 0, 0] = 0
    for s in range(2, n + m + 1):  # cells on anti-diagonal i + j = s (1-based) depend only on earlier diagonals
        i = torch.arange(max(1, s - m), min(n, s - 1) + 1, device=cost.device)
        j = s - i
        previous = torch.minimum(torch.minimum(total[:, i - 1, j], total[:, i, j - 1]), total[:, i - 1, j - 1])
        total[:, i, j] = cost[:, i - 1, j - 1] + previous
    return total[:, n, m] / (n + m)


def dtw_distances(features: np.ndarray, device: str = "cuda", batch_pairs: int = 20_000) -> np.ndarray:
    """DTW distances between all pairs of feature sequences of equal length, shape (N, T, d) -> (N, N)."""
    x = torch.from_numpy(features).float().to(device)
    rows, cols = torch.triu_indices(len(x), len(x), offset=1, device=device)
    distances = torch.zeros(len(x), len(x), device=device)
    for start in range(0, len(rows), batch_pairs):
        r, c = rows[start : start + batch_pairs], cols[start : start + batch_pairs]
        d = dtw_from_cost(torch.cdist(x[r], x[c]))
        distances[r, c], distances[c, r] = d, d
    return distances.cpu().numpy()
