import numpy as np
import pytest

from isolated_sign_validation.landmarks import N_LANDMARKS, LandmarkStore, write_store


def make_clip(clip_id: str, n_frames: int, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    landmarks = rng.random((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    landmarks[0, :21] = np.nan  # a missing hand
    metadata = {"dataset": "test", "clip_id": clip_id, "sign": "hello", "signer": "test:1"}
    return metadata, landmarks


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    clips = [make_clip("a", 3, rng), make_clip("b", 1, rng), make_clip("c", 5, rng)]
    write_store(tmp_path, clips)

    store = LandmarkStore(tmp_path)
    assert len(store) == 3
    assert store.clips["clip_id"].to_list() == ["a", "b", "c"]
    assert store.clips["n_frames"].to_list() == [3, 1, 5]
    for i, (_, landmarks) in enumerate(clips):
        np.testing.assert_array_equal(store[i], landmarks)


def test_rejects_wrong_shape(tmp_path):
    metadata, _ = make_clip("a", 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="bad landmark shape"):
        write_store(tmp_path, [(metadata, np.zeros((2, 10, 3)))])
