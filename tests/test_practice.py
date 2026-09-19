import asyncio
import io

import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from torch import nn

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS
from isolated_sign_validation.practice import create_app, prepare_attempt, sign_means
from isolated_sign_validation.preparation import PrepConfig, mirror, prepare_clip

CONFIG = PrepConfig()
POSE = LANDMARK_SLICES["pose"].start


def attempt_landmarks(hands: tuple[str, ...]) -> np.ndarray:
    """2 s of landmarks with shoulders in view and `hands` raised in the middle second."""
    x = np.full((60, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    x[:, LANDMARK_SLICES["face"]] = [0.5, 0.3, 0.0]
    x[:, LANDMARK_SLICES["pose"]] = [0.5, 0.5, 0.0]
    x[:, POSE + 11, :2] = [0.6, 0.5]
    x[:, POSE + 12, :2] = [0.4, 0.5]
    for hand in hands:
        x[15:45, LANDMARK_SLICES[hand]] = [0.35 if hand == "right_hand" else 0.65, 0.6, 0.0]
    return x


def test_sign_means_give_the_mean_cosine_similarity():
    embeddings = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    means = sign_means(embeddings, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(means @ [1.0, 0.0], [0.5, np.sqrt(0.5)])


@pytest.mark.parametrize(
    "hands, handedness, mirrored",
    [
        (("right_hand",), "left", False),  # one-handed: the detected hand decides
        (("left_hand",), "right", True),
        (("left_hand", "right_hand"), "right", False),  # two-handed: the user's handedness decides
        (("left_hand", "right_hand"), "left", True),
    ],
)
def test_prepare_attempt_mirrors_left_dominant_attempts(hands, handedness, mirrored):
    landmarks = attempt_landmarks(hands)
    frames = prepare_clip(landmarks, 30.0, 1.0, CONFIG)
    expected = mirror(frames, CONFIG) if mirrored else frames
    np.testing.assert_array_equal(prepare_attempt(landmarks, 30.0, 1.0, handedness, CONFIG), expected)


class Fixed(nn.Module):
    """A stand-in for the embedding model: the same embedding for every clip."""

    def forward(self, frames, hands, mask):
        return torch.tensor([[3.0, 4.0]]).expand(len(frames), 2)


def test_attempt_scores_against_the_chosen_sign():
    references = {"A": ["a1"], "B": ["b1", "b2"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])  # A: cosine 1 with the model's embedding, B: 0.6
    app = create_app(references, means, {}, Fixed(), CONFIG, threshold=0.7, device="cpu")
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    def send(landmarks: np.ndarray, sign: str) -> dict:
        upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
        return asyncio.run(attempt(upload, sign=sign, handedness="right", width=640, height=480))

    landmarks = attempt_landmarks(("right_hand",))
    assert send(landmarks, "A") == {"usable": True, "note": "Looks good.", "score": pytest.approx(1.0), "threshold": 0.7, "correct": True}
    assert send(landmarks, "B")["correct"] is False and send(landmarks, "B")["score"] == pytest.approx(0.6)
    assert send(attempt_landmarks(()), "A")["usable"] is False
    with pytest.raises(HTTPException):
        send(landmarks, "C")
    with pytest.raises(HTTPException):
        send(landmarks[:, :-1], "A")  # not a whole number of frames
