import numpy as np
import polars as pl
import pytest

from isolated_sign_validation.landmarks import LANDMARK_GROUPS, N_LANDMARKS, LandmarkStore, write_store


def test_landmark_groups_are_disjoint_and_in_range():
    sizes = {name: len(indices) for name, indices in LANDMARK_GROUPS.items()}
    assert sizes == {"left_hand": 21, "right_hand": 21, "upper_body": 12, "face_reference": 6, "lips": 40}
    indices = np.concatenate(list(LANDMARK_GROUPS.values()))
    assert len(np.unique(indices)) == len(indices)
    assert indices.min() >= 0 and indices.max() < N_LANDMARKS


def make_clip(clip_id: str, n_frames: int, rng: np.random.Generator, fps: float | None = 30.0) -> tuple[dict, np.ndarray]:
    landmarks = rng.random((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    landmarks[0, :21] = np.nan  # a missing hand
    metadata = {"dataset": "test", "clip_id": clip_id, "sign": "hello", "signer": "test:1", "fps": fps}
    return metadata, landmarks


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    clips = [make_clip("a", 3, rng), make_clip("b", 1, rng, fps=None), make_clip("c", 5, rng, fps=25.0)]
    write_store(tmp_path, clips)

    store = LandmarkStore(tmp_path)
    assert len(store) == 3
    assert store.clips["clip_id"].to_list() == ["a", "b", "c"]
    assert store.clips["n_frames"].to_list() == [3, 1, 5]
    assert store.clips["fps"].to_list() == [30.0, None, 25.0]
    for i, (_, landmarks) in enumerate(clips):
        np.testing.assert_array_equal(store[i], landmarks)


def test_unknown_fps_is_stored_as_float(tmp_path):
    write_store(tmp_path, [make_clip("a", 2, np.random.default_rng(0), fps=None)])
    assert LandmarkStore(tmp_path).clips["fps"].dtype == pl.Float64


def test_rejects_wrong_shape(tmp_path):
    metadata, _ = make_clip("a", 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="bad landmark shape"):
        write_store(tmp_path, [(metadata, np.zeros((2, 10, 3)))])
