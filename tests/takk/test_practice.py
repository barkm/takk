import asyncio
import io

import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from torch import nn

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS
from isolated_sign_validation.preparation import PrepConfig, mirror, prepare_clip
from takk.practice import NOTES, create_app, prepare_attempt, sign_means
from takk.vocabulary import spoken_word
from takk.speech import SAMPLE_RATE

from test_speech import wav

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
    app = create_app(references, means, {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: [])
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    def send(landmarks: np.ndarray, *signs: str) -> dict:
        upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
        return asyncio.run(attempt(upload, sign=list(signs), handedness="right", width=640, height=480))

    landmarks = attempt_landmarks(("right_hand",))
    closest = {"sign": "A", "score": pytest.approx(1.0)}
    a = {"sign": "A", "usable": True, "note": "Det ser bra ut.", "score": pytest.approx(1.0), "correct": True, "closest": closest}
    assert send(landmarks, "A") == {"threshold": 0.7, "note": "", "split": "whole", "signs": [a]}
    (b,) = send(landmarks, "B")["signs"]
    assert b["closest"] == closest and b["correct"] is False and b["score"] == pytest.approx(0.6)
    assert send(attempt_landmarks(()), "A")["signs"][0]["usable"] is False
    with pytest.raises(HTTPException):
        send(landmarks, "C")
    with pytest.raises(HTTPException):
        send(landmarks[:, :-1], "A")  # not a whole number of frames


def test_spoken_word_is_the_sign_name_without_its_entry():
    assert spoken_word("sts:platta slag-25563") == "platta slag"
    assert spoken_word("sts:höger-04788") == "höger"


def test_attempt_splits_a_spoken_sentence_by_its_words():
    references = {"A": ["a1"], "B": ["b1"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])
    spoken = [(0.5, 0.7), (2.5, 2.7)]  # "A" then "B", so the cut falls 1.6 s into the audio
    app = create_app(references, means, {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: spoken)
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)  # 4 s, a sign in each half
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, sign=["A", "B"], handedness="right", width=640, height=480, audio=audio, audio_offset=0.1))  # fmt: skip
    assert result["split"] == "speech"
    assert [s["sign"] for s in result["signs"]] == ["A", "B"] and [s["correct"] for s in result["signs"]] == [True, False]


def test_attempt_needs_the_microphone_for_a_sentence_but_not_for_one_sign():
    """The spoken words are the only thing that splits a sentence, so a sentence without audio is
    refused rather than split some other way; one sign is the whole recording and needs no audio."""
    app = create_app({"A": ["a1"]}, np.array([[0.6, 0.8]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: [])
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = lambda: UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))  # noqa: E731

    sentence = asyncio.run(attempt(upload(), sign=["A", "A"], handedness="right", width=640, height=480))
    assert sentence["signs"] == [] and sentence["note"] == NOTES["no_audio"]

    one = asyncio.run(attempt(upload(), sign=["A"], handedness="right", width=640, height=480))
    assert one["split"] == "whole" and [s["sign"] for s in one["signs"]] == ["A"]
