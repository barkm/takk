"""Practicing signs: pick a sign of a glossary, sign it in front of the webcam, and learn whether it was that sign.

This is the app's API only (`main.py`); the page is a separate site, built from `frontend/` and never
served from here (see the decisions in ROADMAP-takk.md). The browser extracts the landmarks itself,
with the same HolisticLandmarker setup as `extraction.py` (see the browser extraction findings in
ROADMAP.md), and sends only those: the video never leaves the user's device. The backend
checks and prepares the attempt like any recording, embeds it, and scores it against the glossary
clips of the chosen sign: the mean cosine similarity to them, as `evaluation` scores a sign from k
references. The attempt counts as the sign when the score reaches a global threshold. The sign of
the whole glossary with the highest score is reported too, so a wrong attempt shows what it resembled.

An attempt can also be a sentence of several signs, as TAKK signs the key words of a spoken
sentence. Every attempt is spoken, one sign or many: the signer says a whole Swedish sentence, its
key words are timed in the audio, and each part of the recording is scored as an attempt of the sign
at its place. The microphone is therefore always needed. Splitting at the rests between the signs
instead was removed (see ROADMAP-takk.md step 3): it could not tell a sign the signer skipped from a
sign it had failed to find, so it voided the whole sentence and with it the verdicts on the signs
that were right, while the speech split scores a skipped sign as a miss.
"""

import anthropic
import numpy as np
import torch
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from torch import nn

from isolated_sign_validation.checks import check_clip
from isolated_sign_validation.dataset import collate
from isolated_sign_validation.extraction import MODEL_PATH, VideoInfo
from isolated_sign_validation.landmarks import N_LANDMARKS, SKELETON_EDGES
from isolated_sign_validation.preparation import ONE_HANDED, PrepConfig, hand_presence, hide_low_hands, mirror, prepare_clip
from takk.speech import MIN_WORD_SCORE, Aligner, decode_audio, split_speech
from takk.story import write_story
from takk.vocabulary import Index, search, spoken_word

# Everything the learner reads is Swedish (see ROADMAP-takk.md): the app is for practising TAKK.
CLOSEST = 20  # signs a search by signing answers with, as many as a list of rows can show at once

NOTES = {
    "empty": "Inspelningen är tom.",
    "recorded": "Inspelat.",
    "no_hands": "Inga händer syntes. Har du händerna i bild när du tecknar?",
    "short": "Bara {seconds:.1f} s tecknande syntes. Spela in igen, lite långsammare.",
    "long": "{seconds:.0f} s tecknande syntes, vilket är för långt för ett tecken.",
    "no_body": "Din överkropp syntes inte. Sitt så att båda axlarna är i bild.",
    "lost_hand": "Går att bedöma, men en hand tappades i {lost:.0%} av bildrutorna medan du tecknade.",
    "ok": "Det ser bra ut.",
    "no_audio": "Mikrofonen behövs: orden du säger är det som visar var tecknen är i inspelningen.",
    "not_said": "Meningens ord hittades inte i det du sa. Säg vart och ett av dem tydligt.",
    "not_heard": "Hörde inte {words}. Säg hela meningen högt medan du tecknar den.",
}

def sign_means(embeddings: np.ndarray, labels: np.ndarray, n_signs: int) -> np.ndarray:
    """The mean of each sign's unit-length clip embeddings, shape (n_signs, dim). Its dot product with
    a unit-length embedding is the mean cosine similarity to the sign's clips."""
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sums = np.zeros((n_signs, unit.shape[1]))
    np.add.at(sums, labels, unit)
    return sums / np.bincount(labels, minlength=n_signs)[:, None]


def prepare_attempt(landmarks: np.ndarray, fps: float, aspect: float, handedness: str, config: PrepConfig) -> np.ndarray | None:
    """Prepare an attempt's landmarks as `prepare_store` prepares a clip, mirrored when its dominant
    hand is the left: a one-handed attempt's detected hand, a two-handed one's the user's
    `handedness` (`add_dominant_hands`, with the signer's handedness stated rather than inferred)."""
    frames = prepare_clip(landmarks, fps, aspect, config)
    if frames is None:
        return None
    present = hand_presence(hide_low_hands(landmarks, aspect, config.max_hand_y))
    left, right, both = present[:, 0].sum(), present[:, 1].sum(), present.all(axis=1).sum()
    if both < ONE_HANDED * (left + right - both):
        dominant = "left" if left > right else "right"
    else:
        dominant = handedness
    return mirror(frames, config) if dominant == "left" else frames


@torch.inference_mode()
def embed_clip(model: nn.Module, frames: np.ndarray, config: PrepConfig, device: str) -> np.ndarray:
    """The unit-length embedding of one prepared clip."""
    hands = np.stack([~np.isnan(frames[:, config.group_slices[hand].start, 0]) for hand in ("left_hand", "right_hand")], axis=1)
    batch = collate([{"frames": torch.from_numpy(np.nan_to_num(frames)), "hands": torch.from_numpy(hands), "label": 0}])
    model.eval()
    with torch.autocast(device, dtype=torch.bfloat16):  # as `training.embed`, which embeds the glossary
        embedding = model(batch["frames"].to(device), batch["hands"].to(device), batch["mask"].to(device)).float().cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)


def create_app(
    references: dict[str, list[str]],
    means: np.ndarray,
    reference_videos: dict[str, str],
    model: nn.Module,
    config: PrepConfig,
    threshold: float,
    device: str,
    aligner: Aligner,
    vocabulary: Index | None = None,
    forms: dict[str, str] | None = None,
    writer: anthropic.Anthropic | None = None,
) -> FastAPI:
    """The practice app: the page, the glossary's signs with their clips (`references`, sign ->
    clip ids, in the order of the rows of `means`, see sign_means), the clips' videos (by clip id, see
    video_paths), the extraction model for the browser, and the scoring of attempts. The `aligner`
    times a spoken sentence's words, which is what splits it into its signs. The `writer` writes the
    story a learner signs their way through (see `story.py`)."""
    app = FastAPI()
    names = list(references)
    index = {sign: i for i, sign in enumerate(names)}

    @app.get("/api/signs")
    def signs() -> dict:
        return {
            "signs": [{"sign": sign, "references": clips} for sign, clips in references.items()],
            "fps": config.fps,
            "max_seconds": config.max_frames / config.fps,
            "edges": {group: edges.tolist() for group, edges in SKELETON_EDGES.items()},  # to draw the tracked landmarks
        }

    @app.get("/api/search")
    def search_words(q: str) -> dict:
        """The signs a learner searching for `q` is offered, a word or a theme alike
        (`vocabulary.search`): each with the sign that scores it, its lexicon entry and the word to
        show when the sign is not named for it. This is how vocabulary grows, so it is the one way in
        (step 9 of ROADMAP-takk.md)."""
        return {"words": search(vocabulary, q) if vocabulary else []}

    @app.post("/api/story")
    def story(words: list[str] = Body(embed=True), parts: int = Body(embed=True)) -> dict:
        """A Swedish story over `words` in `parts` parts, each part with the words it uses in the
        order they are spoken (`story.py`). The parts are empty when there is no writer or it could
        not write one, and the caller then has nothing to tell and says so.

        The words are the learner's own: what the cards say, not the names of the signs that score
        them."""
        told = write_story(writer, words, parts) if writer else None
        return {"parts": [{"text": text, "words": used} for text, used in told or []]}

    @app.get("/api/form/{entry_id}")
    def form(entry_id: str) -> dict:
        """How the lexicon describes the form of an entry's sign, in Swedish ("Flata handen,
        vänsterriktad och inåtvänd, kontakt med bröstet, ..."). It is what a learner is taught by
        besides the clip, so it is fetched per sign rather than sent with the whole glossary."""
        return {"form": (forms or {}).get(entry_id, "")}

    @app.get("/api/reference/{clip_id}")
    def reference(clip_id: str) -> FileResponse:
        if clip_id not in reference_videos:
            raise HTTPException(404, "unknown reference clip")
        return FileResponse(reference_videos[clip_id])

    @app.get("/api/model")
    def extraction_model() -> FileResponse:
        return FileResponse(MODEL_PATH)

    def judge(values: np.ndarray, sign: str, handedness: str, width: int, height: int) -> dict:
        """Score the landmarks of one sign as an attempt of `sign`."""
        usable, note, _ = check_clip(values, VideoInfo(config.fps, width, height), config, notes=NOTES)
        frames = prepare_attempt(values, config.fps, width / height, handedness, config) if usable else None
        if frames is None:
            return {"sign": sign, "usable": False, "note": note}
        scores = means @ embed_clip(model, frames, config, device)
        score, closest = float(scores[index[sign]]), int(np.argmax(scores))
        return {
            "sign": sign, "usable": True, "note": note, "score": score, "correct": score >= threshold,
            "closest": {"sign": names[closest], "score": float(scores[closest])},
        }  # fmt: skip

    @app.post("/api/search")
    async def search_by_sign(landmarks: UploadFile, handedness: str = Form(), width: int = Form(), height: int = Form()) -> dict:  # fmt: skip
        """The lexicon signs closest to a recording of one sign, the nearest first: a learner who
        knows a sign but not its Swedish word finds it by signing it (step 10 of ROADMAP-takk.md).

        This is a lookup and not an attempt, so nothing is spoken and nothing is scored against a
        threshold; the recording is checked and prepared exactly as an attempt is, and the ranking is
        the one `judge` already computes over the whole glossary.
        """
        values = np.frombuffer(await landmarks.read(), dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or values.size % (N_LANDMARKS * 3):
            raise HTTPException(400, "malformed recording")
        values = values.reshape(-1, N_LANDMARKS, 3)
        usable, note, _ = check_clip(values, VideoInfo(config.fps, width, height), config, notes=NOTES)
        frames = prepare_attempt(values, config.fps, width / height, handedness, config) if usable else None
        if frames is None:
            return {"words": [], "note": note}
        scores = means @ embed_clip(model, frames, config, device)
        closest = np.argsort(scores)[::-1][:CLOSEST]
        # a word per sign, as the text search answers with, so both fill the same list of rows
        words = [vocabulary.by_sign[names[at]] for at in closest if vocabulary and names[at] in vocabulary.by_sign]
        return {"words": words, "note": note}

    @app.post("/api/attempt")
    async def attempt(landmarks: UploadFile, sign: list[str] = Form(), handedness: str = Form(), width: int = Form(), height: int = Form(), spoken: list[str] = Form([]), audio: UploadFile | None = None, audio_offset: float = Form(0.0), unheard_is_miss: bool = Form(False)) -> dict:  # fmt: skip
        """Score an attempt of one sign or a sentence of several (`sign` repeated, in order): its
        landmarks as float32 (n_frames, N_LANDMARKS, 3), NaN where not detected, at the preparation's
        frame rate, from frames of `width` x `height` pixels.

        Every attempt is spoken, which is how TAKK is used, so `audio` is required (recorded
        `audio_offset` seconds before the first frame): the signs are located by their words timed in
        it, and scored only when every word was heard. For a single sign that leaves the whole
        recording, so the alignment's only job there is to say the word was said at all.

        `spoken` is the word said for each sign, when that is not the sign's own name: a learner
        practising "blå" signs `sts:öga-02636`, since blå and öga are one sign form, and says "blå".
        Without it each sign is listened for under its own name.

        A word that was not heard refuses the whole attempt, since its span is then arbitrary and the
        signs around it are cut by it. `unheard_is_miss` scores it as a miss of that sign instead and
        keeps the rest of the verdicts, which is what a story needs: it never stops, and a word left
        unsaid is a word left unsigned (step 12 of ROADMAP-takk.md)."""
        if any(s not in index for s in sign):
            raise HTTPException(404, "unknown sign")
        values = np.frombuffer(await landmarks.read(), dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or values.size % (N_LANDMARKS * 3):
            raise HTTPException(400, "malformed attempt")
        values = values.reshape(-1, N_LANDMARKS, 3)
        if spoken and len(spoken) != len(sign):
            raise HTTPException(400, "a spoken word per sign, or none at all")
        if audio is None:
            return {"threshold": threshold, "note": NOTES["no_audio"], "signs": []}
        words = list(spoken) or [spoken_word(s) for s in sign]
        spans = aligner(decode_audio(await audio.read()), words)
        if spans is None:
            return {"threshold": threshold, "note": NOTES["not_said"], "signs": []}
        # The alignment is a Viterbi path and always returns one, so silence aligns as readily as
        # speech; the score is what says the words were spoken at all (see takk.speech).
        unheard = [word for word, (_, _, score) in zip(words, spans) if score < MIN_WORD_SCORE]
        if unheard:
            print(f"an attempt was refused, the words scoring {[round(s, 2) for *_, s in spans]}")  # while MIN_WORD_SCORE is provisional
        if unheard and not unheard_is_miss:
            return {"threshold": threshold, "note": NOTES["not_heard"].format(words=" och ".join(unheard)), "signs": []}  # fmt: skip
        parts = split_speech(spans, audio_offset, len(values), config.fps)
        if len(parts) != len(sign) or any(part.stop - part.start < 2 for part in parts):
            return {"threshold": threshold, "note": NOTES["not_said"], "signs": []}
        signs = [judge(values[part], s, handedness, width, height) for part, s in zip(parts, sign)]
        for judged, word, (_, _, score) in zip(signs, words, spans):
            if score < MIN_WORD_SCORE:  # only with unheard_is_miss: the word was not said, so its sign was not made
                judged.update(correct=False, note=NOTES["not_heard"].format(words=word))
        return {"threshold": threshold, "note": "", "signs": signs}

    return app
