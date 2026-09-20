import asyncio
import io

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from torch import nn

from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS
from isolated_sign_validation.preparation import PrepConfig, mirror, prepare_clip
from takk.practice import NOTES, create_app, prepare_attempt, sign_means
from takk.sentences import Written
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
    heard = [(0.5, 0.7, -0.4)]  # one word, clearly said, so the whole recording is the attempt
    app = create_app(references, means, {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: heard)
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    def send(landmarks: np.ndarray, *signs: str) -> dict:
        upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
        audio = UploadFile(io.BytesIO(wav(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))))
        return asyncio.run(attempt(upload, sign=list(signs), spoken=[], handedness="right", width=640, height=480, audio=audio, audio_offset=0.0))  # fmt: skip

    landmarks = attempt_landmarks(("right_hand",))
    closest = {"sign": "A", "score": pytest.approx(1.0)}
    a = {"sign": "A", "usable": True, "note": "Det ser bra ut.", "score": pytest.approx(1.0), "correct": True, "closest": closest}
    assert send(landmarks, "A") == {"threshold": 0.7, "note": "", "signs": [a]}
    (b,) = send(landmarks, "B")["signs"]
    assert b["closest"] == closest and b["correct"] is False and b["score"] == pytest.approx(0.6)
    assert send(attempt_landmarks(()), "A")["signs"][0]["usable"] is False
    with pytest.raises(HTTPException):
        send(landmarks, "C")
    with pytest.raises(HTTPException):
        send(landmarks[:, :-1], "A")  # not a whole number of frames


def test_sentence_returns_the_words_in_the_order_they_are_spoken():
    """The words are the caller's own and come back reordered to the sentence: the signing order is
    the spoken order, which is what the attempt is then split by."""
    writer = SimpleNamespace(messages=SimpleNamespace(parse=lambda **kwargs: SimpleNamespace(parsed_output=Written(sentence="Jag vill ha mer mjölk"))))  # fmt: skip
    app = create_app({"A": ["a1"]}, np.array([[1.0, 0.0]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: [], writer=writer)  # fmt: skip
    sentence = next(route for route in app.routes if getattr(route, "path", "") == "/api/sentence").endpoint

    assert sentence(words=["mjölk", "mer"]) == {"sentence": "Jag vill ha mer mjölk", "words": ["mer", "mjölk"]}


def test_sentence_is_empty_without_a_writer():
    """Then the session practises the words one at a time, which needs no model at all."""
    app = create_app({"A": ["a1"]}, np.array([[1.0, 0.0]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: [])  # fmt: skip
    sentence = next(route for route in app.routes if getattr(route, "path", "") == "/api/sentence").endpoint
    assert sentence(words=["mjölk"]) == {"sentence": "", "words": []}


def test_attempt_listens_for_the_word_the_learner_practises_not_the_sign_name():
    """A learner practising "blå" signs sts:öga-02636, because blå and öga are one sign form and the
    lower entry names the class. Aligning the class name would listen for "öga" while they say "blå"."""
    heard: list[list[str]] = []
    app = create_app({"sts:öga-2636": ["a1"]}, np.array([[0.6, 0.8]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: (heard.append(words), [(0.5, 0.7, -0.4)])[1])  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    upload = lambda: UploadFile(io.BytesIO(attempt_landmarks(("right_hand",)).astype(np.float32).tobytes()))  # noqa: E731
    audio = lambda: UploadFile(io.BytesIO(wav(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))))  # noqa: E731

    asyncio.run(attempt(upload(), sign=["sts:öga-2636"], spoken=["blå"], handedness="right", width=640, height=480, audio=audio(), audio_offset=0.0))  # fmt: skip
    asyncio.run(attempt(upload(), sign=["sts:öga-2636"], spoken=[], handedness="right", width=640, height=480, audio=audio(), audio_offset=0.0))  # fmt: skip
    assert heard == [["blå"], ["öga"]]  # without a spoken word the sign is listened for under its own name

    with pytest.raises(HTTPException):  # a word per sign or none at all, never some of them
        asyncio.run(attempt(upload(), sign=["sts:öga-2636", "sts:öga-2636"], spoken=["blå"], handedness="right", width=640, height=480, audio=audio()))  # fmt: skip


def test_spoken_word_is_the_sign_name_without_its_entry():
    assert spoken_word("sts:platta slag-25563") == "platta slag"
    assert spoken_word("sts:höger-04788") == "höger"


def test_attempt_splits_a_spoken_sentence_by_its_words():
    references = {"A": ["a1"], "B": ["b1"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])
    spoken = [(0.5, 0.7, -0.4), (2.5, 2.7, -0.3)]  # "A" then "B", so the cut falls 1.6 s into the audio
    app = create_app(references, means, {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: spoken)
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)  # 4 s, a sign in each half
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, audio=audio, audio_offset=0.1))  # fmt: skip
    assert [s["sign"] for s in result["signs"]] == ["A", "B"] and [s["correct"] for s in result["signs"]] == [True, False]


def test_attempt_refuses_a_sentence_whose_words_were_not_spoken():
    """Forced alignment always returns a path, so silence aligns too, only with a poor score. Without
    the score the sentence would be cut at arbitrary places and scored as if the words were heard."""
    silent = [(0.5, 0.7, -5.4), (2.5, 2.7, -5.2)]  # the scores silence gets from the real model
    app = create_app({"A": ["a1"], "B": ["b1"]}, np.array([[0.6, 0.8], [1.0, 0.0]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: silent)  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, audio=audio, audio_offset=0.0))  # fmt: skip
    assert result["signs"] == [] and result["note"] == NOTES["not_heard"].format(words="A och B")


def test_attempt_needs_the_microphone_however_few_signs_it_has():
    """Every attempt is spoken, so the words said over it are the only thing that locates its signs.
    One sign is no exception: the alignment has nothing to cut there, and its job is to say the word
    was said at all."""
    app = create_app({"A": ["a1"]}, np.array([[0.6, 0.8]]), {}, Fixed(), CONFIG, 0.7, "cpu", aligner=lambda audio, words: [])
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = lambda: UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))  # noqa: E731

    sentence = asyncio.run(attempt(upload(), sign=["A", "A"], spoken=[], handedness="right", width=640, height=480))
    assert sentence["signs"] == [] and sentence["note"] == NOTES["no_audio"]

    alone = asyncio.run(attempt(upload(), sign=["A"], spoken=[], handedness="right", width=640, height=480))
    assert alone["signs"] == [] and alone["note"] == NOTES["no_audio"]
