import numpy as np
import polars as pl
import pytest
import torch

from isolated_sign_validation.dataset import AugmentConfig, SignDataset, affine, augment, collate, drop_frames, drop_hand
from isolated_sign_validation.preparation import PrepConfig, PreparedData

CONFIG = PrepConfig()
HANDS = [CONFIG.group_slices["left_hand"], CONFIG.group_slices["right_hand"]]
N_LANDMARKS = len(CONFIG.landmarks)


def clip_frames(n: int, value: float = 1.0, left_hand: bool = False) -> np.ndarray:
    frames = np.full((n, N_LANDMARKS, 2), value, dtype=np.float32)
    if not left_hand:
        frames[:, HANDS[0]] = np.nan
    return frames


def test_affine_rotates_and_shifts():
    frames = np.array([[[1.0, 0.0]]], dtype=np.float32)
    rotated = affine(frames, np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([0.5, 0.0]))  # 90 degrees
    np.testing.assert_allclose(rotated[0, 0], [0.5, 1.0])


def test_drop_frames_keeps_at_least_one():
    frames = clip_frames(5)
    assert len(drop_frames(frames, np.array([True, False, True, False, False]))) == 2
    assert len(drop_frames(frames, np.zeros(5, dtype=bool))) == 1


def test_drop_hand_blanks_a_stretch():
    dropped = drop_hand(clip_frames(10, left_hand=True), HANDS[0], 2, 5)
    assert np.isnan(dropped[2:5, HANDS[0]]).all()
    assert not np.isnan(dropped[:2, HANDS[0]]).any() and not np.isnan(dropped[5:, HANDS[0]]).any()


def test_augment_without_strength_is_identity():
    none = AugmentConfig(rotation=0, scale=0, shift=0, shear=0, speed=(1, 1), frame_drop=0, hand_drop=0)
    frames = clip_frames(20)
    np.testing.assert_array_equal(augment(frames, np.random.default_rng(0), none, HANDS, 128), frames)


def test_augment_keeps_landmarks_and_missing_hands():
    frames = clip_frames(40)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = augment(frames, rng, AugmentConfig(), HANDS, max_frames=45)
        assert out.shape[1:] == frames.shape[1:] and 1 <= len(out) <= 45
        assert np.isnan(out[:, HANDS[0]]).all()  # a missing hand stays missing


@pytest.fixture
def data(tmp_path) -> PreparedData:
    lengths = [3, 5, 4]
    clips = pl.DataFrame({"sign": ["B", "A", "B"], "signer": ["p1", "p2", "p3"], "split": ["train", "train", "val"], "n_frames": lengths})
    clips = clips.with_columns(offset=pl.col("n_frames").cum_sum() - pl.col("n_frames"))
    frames = np.concatenate([clip_frames(n, value=i, left_hand=(i == 1)) for i, n in enumerate(lengths)])
    path = tmp_path / "prepared"
    path.mkdir()
    np.save(path / "frames.npy", frames)
    clips.write_parquet(path / "clips.parquet")
    (path / "config.json").write_text('{"groups": ["left_hand", "right_hand", "upper_body", "face_reference"]}')
    return PreparedData(path)


def test_dataset_items(data):
    train = SignDataset(data, "train")
    assert len(train) == 2 and train.signs == ["A", "B"]
    first, second = train[0], train[1]
    assert first["label"] == 1 and second["label"] == 0
    assert first["frames"].shape == (3, N_LANDMARKS, 2) and not first["frames"].isnan().any()
    assert (first["frames"][:, HANDS[0]] == 0).all()  # missing hand as zeros
    assert first["hands"].tolist() == [[False, True]] * 3
    assert second["hands"].tolist() == [[True, True]] * 5


def test_collate_pads_and_masks(data):
    batch = collate([SignDataset(data, "train")[i] for i in range(2)])
    assert batch["frames"].shape == (2, 5, N_LANDMARKS, 2)
    assert batch["mask"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert batch["labels"].tolist() == [1, 0]
    assert batch["frames"].dtype == torch.float32


def test_dataset_excludes_signers(data):
    dataset = SignDataset(data, "train", exclude_signers={"p2"})
    assert dataset.positions.tolist() == [0] and dataset.signs == ["B"]


def test_dataset_keeps_clips_without_signer(data):
    data.clips = data.clips.with_columns(signer=pl.Series([None, "p2", "p3"], dtype=pl.String))
    assert SignDataset(data, "train", exclude_signers={"p2"}).positions.tolist() == [0]


def test_dataset_leaves_out_clips_without_split(data):
    data.clips = data.clips.with_columns(split=pl.Series(["train", None, "val"], dtype=pl.String))
    assert SignDataset(data, "train").positions.tolist() == [0]


def test_dataset_only_signs(data):
    dataset = SignDataset(data, "train", only_signs={"A"})
    assert dataset.positions.tolist() == [1] and dataset.signs == ["A"]
