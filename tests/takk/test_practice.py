import asyncio
import io

import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from torch import nn

from isolated_sign_verification.preparation import PrepConfig
from isolated_sign_verification.verifier import References, Verifier
from sign_data.landmarks import LANDMARK_SLICES, N_LANDMARKS
from takk.practice import NOTES, create_app
from takk.vocabulary import Index, spoken_word
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


def times(n: int) -> UploadFile:
    """The upload of `n` frames' times, tracked at the verifier's 30 fps."""
    return UploadFile(io.BytesIO((np.arange(n) / 30).astype(np.float32).tobytes()))


class Fixed(nn.Module):
    """A stand-in for the embedding model: the same embedding for every clip."""

    def forward(self, frames, hands, mask):
        return torch.tensor([[3.0, 4.0]]).expand(len(frames), 2)


def glossary(clips: dict[str, list[str]], means: np.ndarray) -> tuple:
    """The first arguments of `create_app` for signs whose references have the given `means`, scored
    with the Fixed model at a threshold of 0.7."""
    return clips, Verifier(Fixed(), CONFIG, 0.7, {}), References(list(clips), means)


def test_attempt_scores_against_the_chosen_sign():
    references = {"A": ["a1"], "B": ["b1", "b2"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])  # A: cosine 1 with the model's embedding, B: 0.6
    heard = [(0.5, 0.7, -0.4)]  # one word, clearly said, so the whole recording is the attempt
    app = create_app(*glossary(references, means), aligner=lambda audio, words: heard)
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    def send(landmarks: np.ndarray, *signs: str) -> dict:
        upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
        audio = UploadFile(io.BytesIO(wav(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))))
        return asyncio.run(attempt(upload, times(len(landmarks)), sign=list(signs), spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio, audio_offset=0.0))  # fmt: skip

    landmarks = attempt_landmarks(("right_hand",))
    closest = {"sign": "A", "score": pytest.approx(1.0)}
    # a recording with nothing wrong with it carries no note: the app never praises the learner
    a = {"sign": "A", "usable": True, "note": "", "score": pytest.approx(1.0), "correct": True, "closest": closest}
    assert send(landmarks, "A") == {"threshold": 0.7, "note": "", "signs": [a]}
    (b,) = send(landmarks, "B")["signs"]
    assert b["closest"] == closest and b["correct"] is False and b["score"] == pytest.approx(0.6)
    assert send(attempt_landmarks(()), "A")["signs"][0]["usable"] is False
    with pytest.raises(HTTPException):
        send(landmarks, "C")
    with pytest.raises(HTTPException):
        send(landmarks[:, :-1], "A")  # not a whole number of frames
    upload = lambda times: UploadFile(io.BytesIO(np.array(times, dtype=np.float32).tobytes()))  # noqa: E731
    for n, wrong in [(3, [0.0, 0.1]), (3, [0.0, 0.1, 0.05]), (3, [0.0, 0.1, 100.0])]:  # a time per frame, never back, not hours
        landmarks = attempt_landmarks(("right_hand",))[:n]
        with pytest.raises(HTTPException):
            asyncio.run(attempt(UploadFile(io.BytesIO(landmarks.tobytes())), upload(wrong), sign=["A"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[]))  # fmt: skip


def test_attempt_listens_for_the_word_the_learner_practises_not_the_sign_name():
    """A learner practising "blå" signs sts:öga-02636, because blå and öga are one sign form and the
    lower entry names the class. Aligning the class name would listen for "öga" while they say "blå"."""
    heard: list[list[str]] = []
    app = create_app(*glossary({"sts:öga-2636": ["a1"]}, np.array([[0.6, 0.8]])), aligner=lambda audio, words: (heard.append(words), [(0.5, 0.7, -0.4)])[1])  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    landmarks = attempt_landmarks(("right_hand",))
    upload = lambda: UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))  # noqa: E731
    audio = lambda: UploadFile(io.BytesIO(wav(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))))  # noqa: E731

    asyncio.run(attempt(upload(), times(len(landmarks)), sign=["sts:öga-2636"], spoken=["blå"], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio(), audio_offset=0.0))  # fmt: skip
    asyncio.run(attempt(upload(), times(len(landmarks)), sign=["sts:öga-2636"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio(), audio_offset=0.0))  # fmt: skip
    assert heard == [["blå"], ["öga"]]  # without a spoken word the sign is listened for under its own name

    with pytest.raises(HTTPException):  # a word per sign or none at all, never some of them
        asyncio.run(attempt(upload(), times(len(landmarks)), sign=["sts:öga-2636", "sts:öga-2636"], spoken=["blå"], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio()))  # fmt: skip


def test_spoken_word_is_the_sign_name_without_its_entry():
    assert spoken_word("sts:platta slag-25563") == "platta slag"
    assert spoken_word("sts:höger-04788") == "höger"


def test_attempt_splits_a_spoken_sentence_by_its_words():
    references = {"A": ["a1"], "B": ["b1"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])
    spoken = [(0.5, 0.7, -0.4), (2.5, 2.7, -0.3)]  # "A" then "B", so the cut falls 1.6 s into the audio
    app = create_app(*glossary(references, means), aligner=lambda audio, words: spoken)
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)  # 4 s, a sign in each half
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio, audio_offset=0.1))  # fmt: skip
    assert [s["sign"] for s in result["signs"]] == ["A", "B"] and [s["correct"] for s in result["signs"]] == [True, False]


def test_attempt_aligns_only_the_audio_around_the_voice():
    """The page says when it heard the voice, and only the audio around it is aligned: the wait before
    speaking and the silence after are left out, and the words are timed from the trimmed audio."""
    references = {"A": ["a1"], "B": ["b1"]}
    means = np.array([[0.6, 0.8], [1.0, 0.0]])
    spoken = [(0.3, 0.5, -0.4), (2.3, 2.5, -0.3)]  # the words of the test above, trimmed from 0.2 s
    aligned = []
    app = create_app(*glossary(references, means), aligner=lambda audio, words: (aligned.append(len(audio)), spoken)[1])  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[0.5, 2.7], audio=audio, audio_offset=0.1))  # fmt: skip
    assert aligned == [round(2.8 * SAMPLE_RATE)]  # 0.2-3.0 s, VOICE_MARGIN around the voice
    assert [s["sign"] for s in result["signs"]] == ["A", "B"] and [s["correct"] for s in result["signs"]] == [True, False]

def test_attempt_refuses_a_sentence_whose_words_were_not_spoken():
    """Forced alignment always returns a path, so silence aligns too, only with a poor score. Without
    the score the sentence would be cut at arbitrary places and scored as if the words were heard."""
    silent = [(0.5, 0.7, -5.4), (2.5, 2.7, -5.2)]  # the scores silence gets from the real model
    app = create_app(*glossary({"A": ["a1"], "B": ["b1"]}, np.array([[0.6, 0.8], [1.0, 0.0]])), aligner=lambda audio, words: silent)  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio, audio_offset=0.0, unheard_is_miss=False))  # fmt: skip
    assert result["signs"] == [] and result["note"] == NOTES["not_heard"].format(words="A och B")


def test_attempt_scores_an_unheard_word_as_a_miss_when_the_caller_asks_for_it():
    """A story never stops: a word the learner did not say is a miss of that sign, with a verdict of
    its own, and the signs around it are still scored."""
    heard = [(0.5, 0.7, -0.2), (2.5, 2.7, -5.2)]  # the second word was not spoken
    app = create_app(*glossary({"A": ["a1"], "B": ["b1"]}, np.array([[0.6, 0.8], [1.0, 0.0]])), aligner=lambda audio, words: heard)  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint

    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
    audio = UploadFile(io.BytesIO(wav(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))))
    result = asyncio.run(attempt(upload, times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[], audio=audio, audio_offset=0.0, unheard_is_miss=True))  # fmt: skip
    assert result["note"] == "" and len(result["signs"]) == 2
    assert result["signs"][1]["correct"] is False and result["signs"][1]["note"] == NOTES["not_heard"].format(words="B")


def test_attempt_needs_the_microphone_however_few_signs_it_has():
    """Without `cuts` from the page, the words said over an attempt are the only thing that locates its
    signs. One sign is no exception: the alignment has nothing to cut there, and its job is to say the
    word was said at all."""
    app = create_app(*glossary({"A": ["a1"]}, np.array([[0.6, 0.8]])), aligner=lambda audio, words: [])
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)
    upload = lambda: UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))  # noqa: E731

    sentence = asyncio.run(attempt(upload(), times(len(landmarks)), sign=["A", "A"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[]))
    assert sentence["signs"] == [] and sentence["note"] == NOTES["no_audio"]

    alone = asyncio.run(attempt(upload(), times(len(landmarks)), sign=["A"], spoken=[], handedness="right", width=640, height=480, cuts=[], voice=[]))
    assert alone["signs"] == [] and alone["note"] == NOTES["no_audio"]


def test_attempt_signed_in_silence_is_split_where_the_page_cut_it():
    """A learner who signs without speaking lowers the hands between the signs, and the page sends
    where each sign starts and ends instead of the audio."""
    app = create_app(*glossary({"A": ["a1"], "B": ["b1"]}, np.array([[0.6, 0.8], [1.0, 0.0]])), aligner=lambda audio, words: [])  # fmt: skip
    attempt = next(route for route in app.routes if getattr(route, "path", "") == "/api/attempt").endpoint
    landmarks = np.concatenate([attempt_landmarks(("right_hand",))] * 2)  # a sign in each half, hands up 0.5-1.5 s and 2.5-3.5 s
    upload = lambda: UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))  # noqa: E731

    result = asyncio.run(attempt(upload(), times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[0.25, 1.75, 2.25, 3.75], voice=[]))  # fmt: skip
    assert result["note"] == "" and [s["correct"] for s in result["signs"]] == [True, False]
    with pytest.raises(HTTPException):
        asyncio.run(attempt(upload(), times(len(landmarks)), sign=["A", "B"], spoken=[], handedness="right", width=640, height=480, cuts=[0.25, 1.75], voice=[]))  # fmt: skip


def test_search_answers_from_the_vocabulary_it_was_given():
    # the endpoint searches the vocabulary passed to `create_app`, which a local name inside it once
    # shadowed, so every search answered 500 until this test existed
    words = [{"sign": "sts:hej-1", "id": "1"}]
    vocabulary = Index(words, {"Hälsningsfras": words}, {"sts:hej-1": words[0]})
    app = create_app(*glossary({"sts:hej-1": ["a1"]}, np.array([[1.0, 0.0]])), aligner=lambda audio, words: [], vocabulary=vocabulary)  # fmt: skip
    search = next(route for route in app.routes if getattr(route, "path", "") == "/api/search").endpoint

    assert search("hej") == {"words": words}
    assert search("hälsning") == {"words": words}  # the theme its name begins
    assert search("x") == {"words": []}


def test_search_by_signing_ranks_the_whole_glossary():
    # the model embeds every recording as [3, 4] / 5, so B is the nearer of the two signs
    references = {"sts:hej-1": ["a1"], "sts:mamma-2": ["b1"]}
    means = np.array([[1.0, 0.0], [0.6, 0.8]])
    words = {"sts:hej-1": {"sign": "sts:hej-1", "id": "1"}, "sts:mamma-2": {"sign": "sts:mamma-2", "id": "2"}}
    vocabulary = Index(list(words.values()), {}, words)
    app = create_app(*glossary(references, means), aligner=lambda audio, words: [], vocabulary=vocabulary)  # fmt: skip
    search = next(route for route in app.routes if getattr(route, "path", "") == "/api/search" and "POST" in route.methods).endpoint  # fmt: skip

    def send(landmarks: np.ndarray) -> dict:
        upload = UploadFile(io.BytesIO(landmarks.astype(np.float32).tobytes()))
        return asyncio.run(search(upload, times(len(landmarks)), handedness="right", width=640, height=480))

    found = send(attempt_landmarks(("right_hand",)))
    assert [word["sign"] for word in found["words"]] == ["sts:mamma-2", "sts:hej-1"]

    # a recording with no hands in it is no lookup at all, and says why rather than ranking noise
    empty = send(np.full((60, N_LANDMARKS, 3), np.nan, dtype=np.float32))
    assert empty["words"] == [] and empty["note"] == NOTES["no_hands"]


def test_the_glossary_is_compressed():
    # 2.3 MB of JSON on every page load, a tenth of that compressed
    app = create_app(*glossary({"A": ["a1"] * 500}, np.array([[1.0, 0.0]])), aligner=lambda audio, words: [])  # fmt: skip
    response = TestClient(app).get("/api/signs", headers={"accept-encoding": "gzip"})

    assert response.headers["content-encoding"] == "gzip"


def test_the_page_may_call_from_another_origin():
    # the page is served from another origin than this API, so a call without CORS never arrives
    app = create_app(*glossary({"A": ["a1"]}, np.array([[1.0, 0.0]])), aligner=lambda audio, words: [])  # fmt: skip
    response = TestClient(app).get("/api/signs", headers={"Origin": "https://example.github.io"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
