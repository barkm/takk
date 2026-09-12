"""Landmark extraction from videos with MediaPipe's HolisticLandmarker.

All video datasets go through this one setup, so their landmarks are consistent with each other.
"""

import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
MODEL_PATH = Path("data/models/holistic_landmarker.task")

# HolisticLandmarkerResult field for each part of the landmark layout
_RESULT_FIELDS = {
    "face": "face_landmarks",
    "left_hand": "left_hand_landmarks",
    "pose": "pose_landmarks",
    "right_hand": "right_hand_landmarks",
}


def download_model(path: Path = MODEL_PATH) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)


def silence_native_logs() -> None:
    """Discard this process's stderr, where MediaPipe's C++ code logs on every model load.

    Meant as initializer for worker processes; exceptions in workers still reach the main process.
    """
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)


def result_to_array(result) -> np.ndarray:
    """Landmarks of one frame (a HolisticLandmarkerResult) in the common layout, shape (N_LANDMARKS, 3).

    Undetected parts are NaN.
    """
    landmarks = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for part, field in _RESULT_FIELDS.items():
        points = getattr(result, field)
        if points:
            s = LANDMARK_SLICES[part]
            # The face model appends 10 iris points to the 468 face mesh points; they are dropped.
            landmarks[s] = [(p.x, p.y, p.z) for p in points[: s.stop - s.start]]
    return landmarks


def extract_landmarks(video: Path, model_path: Path = MODEL_PATH) -> tuple[np.ndarray, float]:
    """Run the HolisticLandmarker over a video.

    Returns the landmarks, shape (n_frames, N_LANDMARKS, 3), and the video's frame rate.
    """
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.VIDEO
    )
    cv2.setNumThreads(1)  # extraction is parallelized over videos; avoid oversubscribing the CPUs
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames, last_timestamp = [], -1
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Video mode requires strictly increasing timestamps, which some containers don't provide.
            timestamp = max(int(capture.get(cv2.CAP_PROP_POS_MSEC)), last_timestamp + 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(result_to_array(landmarker.detect_for_video(image, timestamp)))
            last_timestamp = timestamp
    capture.release()
    return np.stack(frames) if frames else np.empty((0, N_LANDMARKS, 3), dtype=np.float32), fps
