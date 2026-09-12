import numpy as np
import polars as pl

from isolated_sign_validation.datasets.kaggle_asl_signs import read_clip
from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS


def test_read_clip(tmp_path):
    frames = [5, 6, 9]  # gap between 6 and 9
    rows = [
        {
            "frame": frame,
            "type": type_,
            "landmark_index": i,
            # x encodes the frame and y the canonical landmark index; left hand is missing
            "x": None if type_ == "left_hand" else float(frame),
            "y": None if type_ == "left_hand" else float(s.start + i),
            "z": None if type_ == "left_hand" else 0.5,
        }
        for frame in frames
        for type_, s in LANDMARK_SLICES.items()
        for i in range(s.stop - s.start)
    ]
    path = tmp_path / "clip.parquet"
    pl.DataFrame(rows).sample(fraction=1, shuffle=True, seed=0).write_parquet(path)

    landmarks = read_clip(path)

    assert landmarks.shape == (3, N_LANDMARKS, 3)
    assert landmarks.dtype == np.float32
    left_hand = LANDMARK_SLICES["left_hand"]
    assert np.isnan(landmarks[:, left_hand]).all()
    present = np.delete(landmarks, np.r_[left_hand], axis=1)
    np.testing.assert_array_equal(present[..., 0], np.array(frames)[:, None].repeat(present.shape[1], 1))
    expected_index = np.delete(np.arange(N_LANDMARKS), np.r_[left_hand])
    np.testing.assert_array_equal(present[..., 1], np.tile(expected_index, (3, 1)))
    assert (present[..., 2] == 0.5).all()
