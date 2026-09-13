import numpy as np
import pytest
import torch

from isolated_sign_validation.baselines import dtw_distances, dtw_features, dtw_from_cost, hand_embedding
from isolated_sign_validation.preparation import PrepConfig

CONFIG = PrepConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def naive_dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            total[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    return total[n, m] / (n + m)


def test_dtw_matches_naive_implementation():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 7, 3)), rng.normal(size=(2, 5, 3))
    cost = torch.cdist(torch.from_numpy(a), torch.from_numpy(b))
    np.testing.assert_allclose(dtw_from_cost(cost).numpy(), [naive_dtw(a[k], b[k]) for k in range(2)], rtol=1e-6)


def test_dtw_distances_ignore_speed_differences():
    t = np.linspace(0, 1, 32)
    slow = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)
    fast = np.stack([np.sin(2 * np.pi * t**2), np.cos(2 * np.pi * t**2)], axis=1)  # same path, different timing
    other = np.stack([np.cos(4 * np.pi * t), np.sin(4 * np.pi * t)], axis=1)
    d = dtw_distances(np.stack([slow, fast, other]).astype(np.float32), device=DEVICE, batch_pairs=2)
    np.testing.assert_allclose(d, d.T)
    assert (np.diag(d) == 0).all()
    assert d[0, 1] < 0.2 * d[0, 2]


def prepared_frames(n: int, with_left_hand: bool) -> np.ndarray:
    frames = np.random.default_rng(0).normal(size=(n, len(CONFIG.landmarks), 2)).astype(np.float32)
    if not with_left_hand:
        frames[:, CONFIG.group_slices["left_hand"]] = np.nan
    return frames


def test_dtw_features_put_missing_hands_at_the_pose_wrist():
    frames = prepared_frames(40, with_left_hand=False)
    features = dtw_features(frames, CONFIG, n_frames=40).reshape(40, -1, 2)
    left_wrist = frames[:, CONFIG.group_slices["upper_body"].start + 4]
    np.testing.assert_allclose(features[:, :21], np.repeat(left_wrist[:, None], 21, axis=1), rtol=1e-6)
    assert not np.isnan(features).any()


@pytest.mark.parametrize("with_left_hand", [True, False])
def test_hand_embedding_is_finite(with_left_hand):
    embedding = hand_embedding(prepared_frames(20, with_left_hand), CONFIG)
    assert embedding.shape == (21 * 2 + 2 + 8 * 2,) and np.isfinite(embedding).all()
