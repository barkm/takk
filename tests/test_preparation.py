import argparse
import dataclasses
import json

import numpy as np
import polars as pl
import pytest

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS, write_store
from isolated_sign_validation.preparation import (
    PrepConfig,
    PreparedData,
    add_config_arguments,
    add_dominant_hands,
    config_from,
    hide_low_hands,
    mirror,
    prepare_clip,
    prepare_store,
)

CONFIG = PrepConfig()
POSE = LANDMARK_SLICES["pose"].start
RIGHT_HAND = LANDMARK_SLICES["right_hand"]


def position(group: str, i: int = 0) -> int:
    """Position of a group's i-th landmark in the prepared landmark axis."""
    return CONFIG.group_slices[group].start + i


def make_landmarks(n_frames: int, hand_frames) -> np.ndarray:
    """Pose and face at fixed positions, shoulders at x = 0.6 (left) and 0.4 (right), y = 0.5, and a right
    hand in `hand_frames` at x = 0.35 + 0.001 * frame, y = 0.6."""
    x = np.zeros((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    x[:] = [0.5, 0.5, 0.0]
    x[:, LANDMARK_SLICES["left_hand"]] = np.nan
    x[:, RIGHT_HAND] = np.nan
    x[:, POSE + 11, :2] = [0.6, 0.5]
    x[:, POSE + 12, :2] = [0.4, 0.5]
    for t in hand_frames:
        x[t, RIGHT_HAND] = [0.35 + 0.001 * t, 0.6, 0.0]
    return x


def hand_x(frames: np.ndarray, group: str = "right_hand") -> np.ndarray:
    return frames[:, position(group), 0]


def test_excludes_clips_without_hands_or_with_implausible_hand_span():
    assert prepare_clip(make_landmarks(60, []), 30, 1.0, CONFIG) is None
    assert prepare_clip(make_landmarks(60, range(20, 23)), 30, 1.0, CONFIG) is None  # 0.1 s of hands
    assert prepare_clip(make_landmarks(400, range(400)), 30, 1.0, CONFIG) is None  # 13 s of hands


def test_trims_to_hands_with_margin():
    frames = prepare_clip(make_landmarks(60, range(20, 40)), 30, 1.0, CONFIG)
    assert frames.shape == (20 + 2 * 3, len(CONFIG.landmarks), 2)  # 3 frames = 0.1 s margin on each side
    assert np.isnan(hand_x(frames)[:3]).all() and np.isfinite(hand_x(frames)[3:-3]).all()


@pytest.mark.parametrize("aspect", [1.0, 4 / 3, 16 / 9])
def test_normalizes_by_shoulders_with_correct_proportions(aspect):
    frames = prepare_clip(make_landmarks(60, range(20, 40)), 30, aspect, CONFIG)
    np.testing.assert_allclose(frames[5, position("upper_body", 0)], [0.5, 0.0], atol=1e-5)  # left shoulder
    np.testing.assert_allclose(frames[5, position("upper_body", 1)], [-0.5, 0.0], atol=1e-5)  # right shoulder
    # x distances scale with the shoulder width; y is a fraction of the frame height, x of the width
    np.testing.assert_allclose(frames[3, position("right_hand")], [(0.35 + 0.02 - 0.5) / 0.2, 0.1 / (0.2 * aspect)], atol=1e-5)


def test_hides_low_hands():
    landmarks = make_landmarks(60, range(20, 40))
    landmarks[:20, RIGHT_HAND, 1] = 0.75  # resting: 1.25 shoulder widths below the shoulders
    landmarks[40:, RIGHT_HAND] = landmarks[20, RIGHT_HAND] * [1, 0, 1] + [0, 0.65, 0]  # 0.75 below
    hidden = hide_low_hands(landmarks, 1.0, max_hand_y=1.0)
    assert np.isnan(hidden[:20, RIGHT_HAND]).all() and np.isfinite(hidden[20:, RIGHT_HAND]).all()
    np.testing.assert_array_equal(hidden[20:], landmarks[20:])
    # trimmed to the frames where the hand is up
    assert len(prepare_clip(landmarks, 30, 1.0, CONFIG)) == 40 + 3


def test_interpolates_short_hand_gaps_only():
    hand_frames = [t for t in range(20, 50) if not 25 <= t <= 27 and not 33 <= t <= 39]  # gaps of 3 and 7 frames
    x = hand_x(prepare_clip(make_landmarks(60, hand_frames), 30, 1.0, CONFIG))[3:-3]  # frames 20..49
    expected = (0.35 + 0.001 * np.arange(20, 50) - 0.5) / 0.2
    np.testing.assert_allclose(x[5:8], expected[5:8], rtol=1e-5)  # short gap filled linearly
    assert np.isnan(x[13:20]).all()  # long gap stays missing


def test_resamples_to_target_frame_rate_and_caps_length():
    assert len(prepare_clip(make_landmarks(40, range(40)), 15, 1.0, CONFIG)) == 80
    assert len(prepare_clip(make_landmarks(120, range(120)), 60, 1.0, CONFIG)) == 60
    assert len(prepare_clip(make_landmarks(290, range(290)), 30, 1.0, CONFIG)) == CONFIG.max_frames


def test_mirror_swaps_hands_and_flips_x():
    frames = prepare_clip(make_landmarks(60, range(20, 40)), 30, 1.0, CONFIG)
    mirrored = mirror(frames, CONFIG)
    np.testing.assert_allclose(hand_x(mirrored, "left_hand"), -hand_x(frames, "right_hand"))
    assert np.isnan(hand_x(mirrored, "right_hand")).all()
    np.testing.assert_array_equal(mirrored[:, position("upper_body", 0)], frames[:, position("upper_body", 1)] * [-1, 1])
    np.testing.assert_array_equal(mirror(mirrored, CONFIG), frames)


def test_config_requires_both_sides():
    with pytest.raises(ValueError):
        PrepConfig(groups=("right_hand",))


def test_dominant_hand_per_clip_or_signer():
    clips = pl.DataFrame(
        [
            # signer a is left-handed: most one-handed clips show the left hand
            {"signer": "a", "left_frames": 20, "right_frames": 0, "both_frames": 0},
            {"signer": "a", "left_frames": 18, "right_frames": 1, "both_frames": 1},
            {"signer": "a", "left_frames": 0, "right_frames": 20, "both_frames": 0},  # one-handed with the right hand
            {"signer": "a", "left_frames": 20, "right_frames": 25, "both_frames": 18},  # two-handed
            {"signer": "b", "left_frames": 20, "right_frames": 20, "both_frames": 20},  # two-handed, no one-handed clips
            # unknown signers: one-handed clips by their hand, two-handed ones right
            {"signer": None, "left_frames": 20, "right_frames": 0, "both_frames": 0},
            {"signer": None, "left_frames": 20, "right_frames": 20, "both_frames": 20},
        ]
    )
    assert add_dominant_hands(clips)["dominant"].to_list() == ["left", "left", "right", "left", "right", "left", "right"]


def write_prepared(path, signs: list[str], lengths: list[int], value: float, config: PrepConfig = CONFIG) -> None:
    path.mkdir()
    clips = pl.DataFrame({"sign": signs, "n_frames": lengths}).with_columns(offset=pl.col("n_frames").cum_sum() - pl.col("n_frames"))
    clips.write_parquet(path / "clips.parquet")
    np.save(path / "frames.npy", np.full((sum(lengths), len(config.landmarks), config.n_coords), value, dtype=np.float32))
    (path / "config.json").write_text(json.dumps(dataclasses.asdict(config)))


def test_prepared_data_combines_directories(tmp_path):
    write_prepared(tmp_path / "a", ["x", "y"], [3, 2], 1.0)
    write_prepared(tmp_path / "b", ["z"], [4], 2.0)
    data = PreparedData(tmp_path / "a", tmp_path / "b")
    assert data.clips["sign"].to_list() == ["x", "y", "z"]
    assert [len(data[i]) for i in range(3)] == [3, 2, 4]
    assert (data[1] == 1.0).all() and (data[2] == 2.0).all()
    assert data.twins == set()

    (tmp_path / "b" / "twins.csv").write_text("sign_a,sign_b\nx,z\n")
    assert PreparedData(tmp_path / "a", tmp_path / "b").twins == {("x", "z")}

    write_prepared(tmp_path / "c", ["w"], [1], 3.0, dataclasses.replace(CONFIG, fps=25.0))
    with pytest.raises(ValueError):
        PreparedData(tmp_path / "a", tmp_path / "c")


def test_prepare_store(tmp_path):
    def clip(clip_id, landmarks, signer):
        return {"dataset": "t", "clip_id": clip_id, "sign": "s", "signer": signer, "fps": 30.0, "width": 640, "height": 480}, landmarks

    left_handed = make_landmarks(60, [])
    left_handed[20:40, LANDMARK_SLICES["left_hand"]] = [0.65, 0.6, 0.0]
    left_handed[:, RIGHT_HAND] = [0.4, 0.8, 0.0]  # resting low throughout: still a one-handed clip
    write_store(
        tmp_path / "store",
        [clip("right", make_landmarks(60, range(20, 40)), "a"), clip("no_hands", make_landmarks(60, []), "a"), clip("left", left_handed, "b")],
    )
    clips = pl.read_parquet(tmp_path / "store" / "clips.parquet").with_row_index("row")
    prepare_store(tmp_path / "store", clips, CONFIG, tmp_path / "prepared", rows_per_task=1)

    data = PreparedData(tmp_path / "prepared")
    assert data.clips["clip_id"].to_list() == ["right", "left"]
    assert data.clips["dominant"].to_list() == ["right", "left"]
    assert data.config == CONFIG
    for i in range(len(data)):  # the dominant hand ends up in the right_hand slot, on the image's left
        assert np.isfinite(hand_x(data[i])[3:-3]).all() and (hand_x(data[i])[3:-3] < 0).all()
        assert np.isnan(hand_x(data[i], "left_hand")).all()


def test_config_options_default_to_the_prep_config():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    assert config_from(parser.parse_args([])) == PrepConfig()
    lips = config_from(parser.parse_args(["--groups", *PrepConfig().groups, "lips", "--n_coords", "3"]), max_hand_y=0.9)
    assert lips == PrepConfig(groups=(*PrepConfig().groups, "lips"), n_coords=3, max_hand_y=0.9)
