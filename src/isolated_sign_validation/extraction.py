"""Landmark extraction from videos with MediaPipe's HolisticLandmarker.

All video datasets go through this one setup, so their landmarks are consistent with each other.
"""

import os
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS, VideoInfo, write_store_resumable
from isolated_sign_validation.parallel import parallel_map

# Holistic extraction uses ~1.7 cores per process; more workers than this are slower.
MAX_WORKERS = 8
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


def extract_landmarks(video: Path, model_path: Path = MODEL_PATH) -> tuple[np.ndarray, VideoInfo]:
    """Run the HolisticLandmarker over a video.

    Returns the landmarks, shape (n_frames, N_LANDMARKS, 3), and the video's frame rate and frame size.
    """
    # MediaPipe and OpenCV are imported here rather than at the top of the module so that reading a
    # dataset's metadata does not need them: the practice app takes landmarks from a browser and
    # extracts none, and its image carries neither.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.VIDEO
    )
    cv2.setNumThreads(1)  # extraction is parallelized over videos; avoid oversubscribing the CPUs
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, last_timestamp = [], -1
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]  # the decoded size is what MediaPipe sees
            # Video mode requires strictly increasing timestamps, which some containers don't provide.
            timestamp = max(int(capture.get(cv2.CAP_PROP_POS_MSEC)), last_timestamp + 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(result_to_array(landmarker.detect_for_video(image, timestamp)))
            last_timestamp = timestamp
    capture.release()
    landmarks = np.stack(frames) if frames else np.empty((0, N_LANDMARKS, 3), dtype=np.float32)
    return landmarks, VideoInfo(fps, width, height)


def extract_clips(dataset: str, videos: Sequence[dict]) -> Iterator[tuple[dict, np.ndarray]]:
    """Extract (metadata, landmarks) for `videos` (rows with clip_id, sign, path and, if the dataset
    has signer ids, signer), in order."""
    results = parallel_map(extract_landmarks, [Path(video["path"]) for video in videos], max_workers=MAX_WORKERS, initializer=silence_native_logs)  # fmt: skip
    for video, (landmarks, info) in zip(videos, results):
        metadata = {
            "dataset": dataset,
            "clip_id": video["clip_id"],
            "sign": video["sign"],
            "signer": video.get("signer"),
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        }
        yield metadata, landmarks


def extract_store(dataset: str, videos: pl.DataFrame, store_dir: Path) -> None:
    """Extract landmarks for `videos` (see extract_clips) into a landmark store, resuming an interrupted run."""
    download_model()
    write_store_resumable(
        store_dir,
        videos.to_dicts(),
        lambda batch: tqdm(extract_clips(dataset, batch), total=len(batch), desc=dataset, unit="clip"),
    )
