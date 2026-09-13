import numpy as np
import polars as pl
import pytest

from isolated_sign_validation.landmarks import (
    LANDMARK_GROUPS,
    MIRROR_INDEX,
    N_LANDMARKS,
    SKELETON_EDGES,
    LandmarkStore,
    merge_stores,
    write_store,
    write_store_resumable,
)


def test_landmark_groups_are_disjoint_and_in_range():
    sizes = {name: len(indices) for name, indices in LANDMARK_GROUPS.items()}
    assert sizes == {"left_hand": 21, "right_hand": 21, "upper_body": 12, "face_reference": 6, "lips": 40}
    indices = np.concatenate(list(LANDMARK_GROUPS.values()))
    assert len(np.unique(indices)) == len(indices)
    assert indices.min() >= 0 and indices.max() < N_LANDMARKS


def make_clip(clip_id: str, n_frames: int, rng: np.random.Generator, fps: float | None = 30.0) -> tuple[dict, np.ndarray]:
    """A random clip; `fps=None` means the video info is unknown."""
    landmarks = rng.random((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    landmarks[0, :21] = np.nan  # a missing hand
    width, height = (640, 480) if fps is not None else (None, None)
    metadata = {"dataset": "test", "clip_id": clip_id, "sign": "hello", "signer": "test:1", "fps": fps, "width": width, "height": height}  # fmt: skip
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
    assert store.clips["width"].to_list() == [640, None, 640]
    for i, (_, landmarks) in enumerate(clips):
        np.testing.assert_array_equal(store[i], landmarks)


def test_unknown_metadata_keeps_column_types(tmp_path):
    metadata, landmarks = make_clip("a", 2, np.random.default_rng(0), fps=None)
    write_store(tmp_path, [(metadata | {"signer": None}, landmarks)])
    schema = LandmarkStore(tmp_path).clips.schema
    assert (schema["signer"], schema["fps"], schema["width"], schema["height"]) == (pl.String, pl.Float64, pl.Int64, pl.Int64)


def test_rejects_wrong_shape(tmp_path):
    metadata, _ = make_clip("a", 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="bad landmark shape"):
        write_store(tmp_path, [(metadata, np.zeros((2, 10, 3)))])


def test_merge_stores(tmp_path):
    rng = np.random.default_rng(0)
    clips = [make_clip("a", 3, rng), make_clip("b", 2, rng), make_clip("c", 4, rng)]
    write_store(tmp_path / "part1", clips[:2])
    write_store(tmp_path / "part2", clips[2:])

    merge_stores([tmp_path / "part1", tmp_path / "part2"], tmp_path / "merged")

    store = LandmarkStore(tmp_path / "merged")
    assert store.clips["clip_id"].to_list() == ["a", "b", "c"]
    assert store.clips["offset"].to_list() == [0, 3, 5]
    for i, (_, landmarks) in enumerate(clips):
        np.testing.assert_array_equal(store[i], landmarks)


def clip_for(item: int) -> tuple[dict, np.ndarray]:
    landmarks = np.full((item % 3 + 1, N_LANDMARKS, 3), item, dtype=np.float32)
    metadata = {"dataset": "test", "clip_id": str(item), "sign": "hello", "signer": "test:1", "fps": 30.0, "width": 640, "height": 480}  # fmt: skip
    return metadata, landmarks


def test_write_store_resumable_resumes_after_interruption(tmp_path):
    items = list(range(10))

    def crashing_convert(batch):
        for item in batch[:5]:  # dies halfway through the second chunk
            yield clip_for(item)
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        write_store_resumable(tmp_path, items, crashing_convert, chunk_size=3)

    converted = []

    def convert(batch):
        converted.extend(batch)
        return map(clip_for, batch)

    write_store_resumable(tmp_path, items, convert, chunk_size=3)

    assert converted == list(range(3, 10))  # the first chunk was not redone
    assert not (tmp_path / "chunks").exists()
    store = LandmarkStore(tmp_path)
    assert store.clips["clip_id"].to_list() == [str(i) for i in items]
    for i in items:
        np.testing.assert_array_equal(store[i], clip_for(i)[1])
    with pytest.raises(FileExistsError):
        write_store_resumable(tmp_path, items, convert, chunk_size=3)


def test_mirror_index_swaps_sides_and_is_its_own_inverse():
    np.testing.assert_array_equal(MIRROR_INDEX[MIRROR_INDEX], np.arange(N_LANDMARKS))
    np.testing.assert_array_equal(MIRROR_INDEX[LANDMARK_GROUPS["left_hand"]], LANDMARK_GROUPS["right_hand"])
    for group in ("upper_body", "face_reference", "lips"):  # groups containing both sides of the body
        assert set(MIRROR_INDEX[LANDMARK_GROUPS[group]]) == set(LANDMARK_GROUPS[group])


def test_mirror_index_maps_skeleton_onto_itself():
    # Mirroring a connected pair of points must give another connected pair.
    for group, edges in SKELETON_EDGES.items():
        mirrored = {tuple(sorted(pair)) for pair in MIRROR_INDEX[edges]}
        partner = {"left_hand": "right_hand", "right_hand": "left_hand"}.get(group, group)
        assert mirrored == {tuple(sorted(pair)) for pair in SKELETON_EDGES[partner]}, group
