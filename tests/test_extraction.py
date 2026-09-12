from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from isolated_sign_validation.extraction import MODEL_PATH, extract_landmarks, result_to_array
from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS


def points(n: int, value: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=value, y=float(i), z=0.0) for i in range(n)]


def test_result_to_array_places_parts_and_drops_iris():
    result = SimpleNamespace(
        face_landmarks=points(478, 1.0),  # 468 face mesh + 10 iris points
        left_hand_landmarks=[],  # not detected
        pose_landmarks=points(33, 2.0),
        right_hand_landmarks=points(21, 3.0),
    )
    landmarks = result_to_array(result)

    assert landmarks.shape == (N_LANDMARKS, 3)
    assert (landmarks[LANDMARK_SLICES["face"], 0] == 1.0).all()
    np.testing.assert_array_equal(landmarks[LANDMARK_SLICES["face"], 1], np.arange(468))
    assert np.isnan(landmarks[LANDMARK_SLICES["left_hand"]]).all()
    assert (landmarks[LANDMARK_SLICES["pose"], 0] == 2.0).all()
    assert (landmarks[LANDMARK_SLICES["right_hand"], 0] == 3.0).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MediaPipe model not downloaded")
def test_extract_landmarks_from_video_without_person(tmp_path):
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 25, (320, 240))
    for _ in range(5):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    landmarks, fps = extract_landmarks(video)

    assert landmarks.shape == (5, N_LANDMARKS, 3)
    assert np.isnan(landmarks).all()
    assert fps == 25
