import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isolated_sign_validation.speech import SAMPLE_RATE, align_words, decode_audio, split_speech, spoken_boundary, word_targets

VOCAB = {"<pad>": 0, "A": 1, "B": 2, "C": 3}


class FakeProcessor:
    """A stand-in for the model's Wav2Vec2Processor over `VOCAB`."""

    tokenizer = SimpleNamespace(pad_token_id=0, get_vocab=lambda: dict(VOCAB))

    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_values=torch.from_numpy(audio)[None])


def fake_model(spoken: list[str]):
    """A stand-in for the CTC model: one character of `VOCAB` spoken clearly in every frame."""
    logits = torch.full((1, len(spoken), len(VOCAB)), -10.0)
    for frame, character in enumerate(spoken):
        logits[0, frame, VOCAB[character]] = 10.0
    return lambda values: SimpleNamespace(logits=logits)


def wav(audio: np.ndarray) -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as file:
        file.setnchannels(1), file.setsampwidth(2), file.setframerate(SAMPLE_RATE)
        file.writeframes((audio * 32767).astype("<i2").tobytes())
    return data.getvalue()


def test_word_targets_put_a_wildcard_between_and_around_the_words():
    targets, spans = word_targets(["ab", "c"], VOCAB)
    star = len(VOCAB)
    assert targets == [star, 1, 2, star, 3, star] and spans == [(1, 2), (4, 4)]
    assert word_targets(["ö"], VOCAB) is None  # no character the model knows


def test_align_words_times_the_known_words_among_unknown_speech():
    # 100 frames of 0.02 s: "a" at 0.4-0.6 s and "b" at 1.2-1.4 s, everything else speech we cannot
    # know, which only the wildcard can match.
    spoken = ["C"] * 20 + ["A"] * 10 + ["C"] * 30 + ["B"] * 10 + ["C"] * 30
    audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    (a_start, a_end), (b_start, b_end) = align_words(audio, ["a", "b"], fake_model(spoken), FakeProcessor(), "cpu")
    assert 0.4 <= a_start < a_end <= 0.6
    assert 1.2 <= b_start < b_end <= 1.4
    assert align_words(audio, ["ö"], fake_model(spoken), FakeProcessor(), "cpu") is None
    assert align_words(audio, list("abcabcabc" * 20), fake_model(spoken), FakeProcessor(), "cpu") is None  # too few frames


def test_split_speech_cuts_halfway_between_the_words():
    spans = [(0.5, 0.7), (1.5, 1.7)]  # the cut falls at 1.1 s, halfway from one word's end to the next
    assert split_speech(spans, 0.0, 90, 30) == [slice(0, 33), slice(33, 90)]
    assert split_speech(spans, 0.2, 90, 30) == [slice(0, 27), slice(27, 90)]  # the audio started earlier
    assert split_speech([(0.5, 0.7)], 0.0, 90, 30) == [slice(0, 90)]


def test_decode_audio_reads_the_browser_recording():
    audio = np.sin(np.arange(SAMPLE_RATE) * 0.1).astype(np.float32)
    decoded = decode_audio(wav(audio))
    assert len(decoded) == SAMPLE_RATE
    np.testing.assert_allclose(decoded, audio, atol=1e-4)


def test_spoken_boundary_waits_until_the_second_word_has_been_said():
    spoken = [(0.5, 0.7), (1.5, 1.7)]  # both said, and 0.3 s of audio follows the second
    assert spoken_boundary(spoken, 2.0) == pytest.approx(1.1)
    assert spoken_boundary(spoken, 1.9) is None  # too close to the end to trust
    crammed = [(0.5, 0.7), (1.95, 2.0)]  # the second word is not said, so it lands at the very end
    assert spoken_boundary(crammed, 2.0) is None
