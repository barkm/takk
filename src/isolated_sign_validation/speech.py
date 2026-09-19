"""Finding the signs of a TAKK sentence by the words the signer speaks.

TAKK is signed while speaking, so the signs of a sentence can be found in the speech instead of in
the rests between them (see `practice.split_signs`): the signer speaks a whole Swedish sentence and
signs its key words. Those key words are known — they are the signs of the sentence — so their times
come from forced alignment of exactly those words to the audio, not from recognizing what was said.

What is spoken between the key words is unknown, so a wildcard token stands between and around them.
It is an extra column of zeros in the emissions, which is the highest possible log probability, so it
matches any audio; only the key words themselves have to be spoken as they are written. The trick,
and the rest of the recipe, is that of KBLab's `easyaligner` (https://github.com/kb-labb/easyaligner,
MIT), which applies the wildcard at the ends of a transcript; the library itself is built for batch
alignment of long recordings and pins an older torch, so only its approach is used here.
"""

import subprocess
from collections.abc import Callable

import numpy as np
import torch
import torchaudio.functional as F
from transformers import AutoModelForCTC, AutoProcessor

# The Swedish wav2vec2 CTC model whose character probabilities the words are aligned against, about
# 1.2 GB, downloaded once. It is KBLab's, as are the Swedish models `easyaligner` is used with.
MODEL = "KBLab/wav2vec2-large-voxrex-swedish"
SAMPLE_RATE = 16000

Aligner = Callable[[np.ndarray, list[str]], list[tuple[float, float]] | None]


def decode_audio(data: bytes) -> np.ndarray:
    """A recording's audio as mono float32 at `SAMPLE_RATE`, from any container ffmpeg reads (the
    browser picks its own, see practice.html). Whatever decoded is returned even when ffmpeg fails,
    since live mode sends a recording that is still being written and so ends mid-stream; too little
    audio to align simply leaves the caller without a boundary."""
    command = ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]  # fmt: skip
    audio = subprocess.run(command, input=data, capture_output=True, check=False).stdout
    return np.frombuffer(audio, dtype=np.float32)


def word_targets(words: list[str], vocab: dict[str, int]) -> tuple[list[int], list[tuple[int, int]]] | None:
    """The alignment targets for `words`, a wildcard between and around them, and the first and last
    target of each word. None when a word has no character the model knows."""
    star = len(vocab)  # the wildcard is appended to the model's characters, see align_words
    targets, spans = [star], []
    for word in words:
        characters = [vocab[character] for character in word.upper() if character in vocab]
        if not characters:
            return None
        spans.append((len(targets), len(targets) + len(characters) - 1))
        targets += [*characters, star]
    return targets, spans


def align_words(audio: np.ndarray, words: list[str], model: torch.nn.Module, processor, device: str) -> list[tuple[float, float]] | None:  # fmt: skip
    """When each of `words` is spoken in `audio`, as (start, end) seconds. None when they cannot be
    aligned: a word of unknown characters, or audio too short for them."""
    targets = word_targets(words, processor.tokenizer.get_vocab())
    if targets is None:
        return None
    targets, spans = targets
    with torch.inference_mode():
        values = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
        emissions = torch.log_softmax(model(values).logits, dim=-1)
    emissions = torch.cat([emissions, torch.zeros_like(emissions[:, :, :1])], dim=2)  # the wildcard
    if emissions.shape[1] < len(targets):
        return None
    alignment, scores = F.forced_align(emissions, torch.tensor([targets], device=device), blank=processor.tokenizer.pad_token_id)  # fmt: skip
    tokens = F.merge_tokens(alignment[0], scores[0])  # one span per target, blanks and repeats merged
    seconds = len(audio) / SAMPLE_RATE / emissions.shape[1]
    return [(tokens[first].start * seconds, tokens[last].end * seconds) for first, last in spans]


def load_aligner(device: str = "cpu") -> Aligner:
    """Load `MODEL` and return the function that times a sentence's words in a recording's audio."""
    model = AutoModelForCTC.from_pretrained(MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    return lambda audio, words: align_words(audio, words, model, processor, device)


# Seconds of audio that must follow a word before it counts as spoken, see spoken_boundary. It is
# also about the right context the model needs, as its emissions are not causal.
TAIL = 0.3


def spoken_boundary(spans: list[tuple[float, float]], duration: float) -> float | None:
    """Where to cut between two key words of a sentence being signed live: halfway from the first
    word's end to the second word's start, once the second word has been spoken. None while it has
    not been. Forced alignment always places a word somewhere, so it cannot report one as missing;
    an unspoken word is crammed against the end of the audio, because the wildcard before it matches
    everything and scores better than any character. A word therefore counts as spoken only once
    `TAIL` seconds of audio follow it."""
    (_, first_end), (second_start, second_end) = spans
    if second_end > duration - TAIL:
        return None
    return (first_end + second_start) / 2


def split_speech(spans: list[tuple[float, float]], offset: float, n_frames: int, fps: float) -> list[slice]:
    """The signs of a recording at the preparation's frame rate, one per spoken word of `spans`, cut
    halfway between one word's end and the next one's start: a sign runs alongside its word but may
    start before it or end after it. `offset` is how far into the audio the first frame was captured."""
    cuts = [(end + start) / 2 for (_, end), (start, _) in zip(spans[:-1], spans[1:])]
    inner = np.maximum.accumulate(np.clip(np.round((np.array(cuts) - offset) * fps), 0, n_frames)).astype(int)
    edges = [0, *inner.tolist(), n_frames]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
