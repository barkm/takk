"""Practicing signs: pick a sign of a glossary, sign it in front of the webcam, and learn whether it was that sign.

This is the app's API only (`main.py`); the page is a separate site, built from `frontend/` and never
served from here. The browser extracts the landmarks itself, with the same HolisticLandmarker setup
as `extraction.py`, and sends only those: the video never leaves the user's device. The verifier
(`isolated_sign_verification.verifier`) checks the attempt and scores it against every sign of the
glossary; it counts as the chosen sign when that sign's score reaches the verifier's threshold. The
sign of the whole glossary with the highest score is reported too, so a wrong attempt shows what it
resembled.

An attempt can also be a sentence of several signs, as TAKK signs the key words of a spoken
sentence. A learner who speaks while signing says a whole Swedish sentence, its key words are timed
in the audio, and each part of the recording is scored as an attempt of the sign at its place; a
word left unsaid can be scored as a miss of its sign while the rest keep their verdicts. A learner
who signs in silence lowers the hands between the signs, and the page splits the recording there
and sends where each sign is; a sentence with the wrong number of signs is signed again, since which
sign was skipped cannot be told.
"""

import os
import time

import anthropic
import numpy as np
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from isolated_sign_verification.verifier import Attempt, References, Verifier
from sign_data.landmarks import N_LANDMARKS
from takk.speech import MIN_WORD_SCORE, SAMPLE_RATE, Aligner, decode_audio, split_speech, trim
from takk.story import write_story
from takk.suggest import suggest
from takk.vocabulary import Index, search, spoken_word

# Everything the learner reads is Swedish: the app is for practising TAKK.
CLOSEST = 20  # signs a search by signing answers with, as many as a list of rows can show at once

NOTES = {
    "empty": "Inspelningen är tom.",
    "recorded": "Inspelat.",
    "no_hands": "Inga händer syntes. Har du händerna i bild när du tecknar?",
    "short": "Bara {seconds:.1f} s tecknande syntes. Spela in igen, lite långsammare.",
    "long": "{seconds:.0f} s tecknande syntes, vilket är för långt för ett tecken.",
    "no_body": "Din överkropp syntes inte. Sitt så att båda axlarna är i bild.",
    "lost_hand": "Går att bedöma, men en hand tappades i {lost:.0%} av bildrutorna medan du tecknade.",
    # A recording with nothing wrong with it says nothing: the app tells the learner what to fix and
    # never praises them, and "usable" already carries that it was fine.
    "ok": "",
    "no_audio": "Mikrofonen behövs: orden du säger är det som visar var tecknen är i inspelningen.",
    "not_said": "Meningens ord hittades inte i det du sa. Säg vart och ett av dem tydligt.",
    "not_heard": "Hörde inte {words}. Säg hela meningen högt medan du tecknar den.",
}

def create_app(
    clips: dict[str, list[str]],
    verifier: Verifier,
    references: References,
    aligner: Aligner,
    vocabulary: Index | None = None,
    forms: dict[str, str] | None = None,
    writer: anthropic.Anthropic | None = None,
) -> FastAPI:
    """The practice app: the glossary's signs (the `verifier`'s `references`) with the addresses
    their clips are watched at (`clips`, sign -> clip URLs, see `bundle.py`), and the scoring of
    attempts. The `aligner` times a spoken sentence's words, which
    is what splits it into its signs. The `writer` writes the story a learner signs their way through
    (see `story.py`)."""
    app = FastAPI()
    # The glossary is 2.3 MB of JSON and every page load fetches it, which compresses to 271 kB: the
    # page loads quicker and the egress is a tenth.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    # The page is served from another origin than this API, so every call from it is cross-origin.
    # Any origin may call: there are no accounts, no cookies and nothing here that is not the public
    # lexicon. `TAKK_ORIGINS` narrows it when that changes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("TAKK_ORIGINS", "*").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    names = references.signs
    index = {sign: i for i, sign in enumerate(names)}
    threshold = verifier.threshold

    @app.get("/api/signs")
    def signs() -> dict:
        """The glossary, each sign with the addresses of its clips: the lexicon's own, so the page
        plays them whether or not this server is up."""
        return {
            "signs": [{"sign": sign, "references": clips.get(sign, [])} for sign in names],
            "max_seconds": verifier.max_seconds,
        }

    @app.get("/api/search")
    def search_words(q: str) -> dict:
        """The signs a learner searching for `q` is offered, a word or a theme alike
        (`vocabulary.search`): each with the sign that scores it, its lexicon entry and the word to
        show when the sign is not named for it. This is how vocabulary grows, so it is the one way
        in."""
        return {"words": search(vocabulary, q) if vocabulary else []}

    @app.post("/api/story")
    def story(words: list[str] = Body(embed=True), parts: int = Body(embed=True), about: str = Body("", embed=True)) -> dict:  # fmt: skip
        """A Swedish story over `words` in about `parts` parts, each part with the signs it is
        practised by: the form the part says (`said`) and the offered word that form signs (`word`),
        in the order they are spoken (`story.py`). The parts are empty when there is no writer or it
        could not write one, and the caller then has nothing to tell and says so.

        The words are the learner's own: what the cards say, not the names of the signs that score
        them, and `about` is how the learner describes themselves on Om dig."""
        told = write_story(writer, words, parts, about) if writer else None
        return {"parts": [part.model_dump() for part in told or []]}

    @app.post("/api/suggest")
    def suggested(level: str = Body(embed=True), about: str = Body("", embed=True), known: list[str] = Body([], embed=True)) -> dict:  # fmt: skip
        """The chips Sök offers under its field (`suggest.py`): a few labelled sets of words fitted to
        the learner's `level` and `about`, leaving out the words in `known`. Empty when there is no
        writer or no vocabulary."""
        return {"chips": suggest(writer, vocabulary, level, about, known) if writer and vocabulary else []}

    @app.get("/api/form/{entry_id}")
    def form(entry_id: str) -> dict:
        """How the lexicon describes the form of an entry's sign, in Swedish ("Flata handen,
        vänsterriktad och inåtvänd, kontakt med bröstet, ..."). It is what a learner is taught by
        besides the clip, so it is fetched per sign rather than sent with the whole glossary."""
        return {"form": (forms or {}).get(entry_id, "")}

    def decode(landmarks: bytes, times: bytes, handedness: str, width: int, height: int) -> np.ndarray:
        """An uploaded recording's landmarks at the verifier's frame rate, shape (n_frames,
        N_LANDMARKS, 3), from the tracked frames and their times."""
        values, at = np.frombuffer(landmarks, dtype=np.float32), np.frombuffer(times, dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or not at.size or values.size != at.size * N_LANDMARKS * 3:  # fmt: skip
            raise HTTPException(400, "malformed recording")
        # at least a tracked frame a second, so a few frames can't claim hours to resample
        if not np.isfinite(at).all() or not (np.diff(at) >= 0).all() or at[-1] - at[0] > at.size:
            raise HTTPException(400, "malformed recording")
        return verifier.at_frame_rate(values.reshape(-1, N_LANDMARKS, 3), at.astype(np.float64))

    def scores_of(values: np.ndarray, handedness: str, width: int, height: int) -> tuple[np.ndarray | None, str]:
        """The recording of one sign scored against every sign of the glossary, or None when it cannot
        be used, and the check's note on it."""
        attempt = Attempt(values, width, height, handedness)
        check = verifier.check(attempt)
        note = NOTES[check.reason].format(**check.details)
        return (verifier.scores(attempt, references) if check.usable else None), note

    def judge(values: np.ndarray, sign: str, handedness: str, width: int, height: int) -> dict:
        """Score the landmarks of one sign as an attempt of `sign`."""
        scores, note = scores_of(values, handedness, width, height)
        if scores is None:
            return {"sign": sign, "usable": False, "note": note}
        score, closest = float(scores[index[sign]]), int(np.argmax(scores))
        return {
            "sign": sign, "usable": True, "note": note, "score": score, "correct": score >= threshold,
            "closest": {"sign": names[closest], "score": float(scores[closest])},
        }  # fmt: skip

    @app.post("/api/search")
    async def search_by_sign(landmarks: UploadFile, times: UploadFile, handedness: str = Form(), width: int = Form(), height: int = Form()) -> dict:  # fmt: skip
        """The lexicon signs closest to a recording of one sign, the nearest first: a learner who
        knows a sign but not its Swedish word finds it by signing it.

        This is a lookup and not an attempt, so nothing is spoken and nothing is scored against a
        threshold; the recording is checked and prepared exactly as an attempt is, and the ranking is
        the one `judge` already computes over the whole glossary.
        """
        scores, note = scores_of(decode(await landmarks.read(), await times.read(), handedness, width, height), handedness, width, height)
        if scores is None:
            return {"words": [], "note": note}
        closest = np.argsort(scores)[::-1][:CLOSEST]
        # a word per sign, as the text search answers with, so both fill the same list of rows
        words = [vocabulary.by_sign[names[at]] for at in closest if vocabulary and names[at] in vocabulary.by_sign]
        return {"words": words, "note": note}

    @app.post("/api/attempt")
    async def attempt(landmarks: UploadFile, times: UploadFile, sign: list[str] = Form(), handedness: str = Form(), width: int = Form(), height: int = Form(), spoken: list[str] = Form([]), audio: UploadFile | None = None, audio_offset: float = Form(0.0), unheard_is_miss: bool = Form(False), cuts: list[float] = Form([]), voice: list[float] = Form([])) -> dict:  # fmt: skip
        """Score an attempt of one sign or a sentence of several (`sign` repeated, in order): its
        landmarks as float32 (n_frames, N_LANDMARKS, 3), NaN where not detected, tracked at `times`
        (float32 seconds, non-decreasing), from frames of `width` x `height` pixels.

        A spoken attempt sends its `audio` (recorded `audio_offset` seconds before the first frame):
        the signs are located by their words timed in it, and scored only when every word was heard.
        For a single sign that leaves the whole recording, so the alignment's only job there is to
        say the word was said at all. Without `audio` or `cuts` nothing locates the signs, and the
        attempt is refused.

        `spoken` is the word said for each sign, when that is not the sign's own name: a learner
        practising "blå" signs `sts:öga-02636`, since blå and öga are one sign form, and says "blå".
        Without it each sign is listened for under its own name.

        A word that was not heard refuses the whole attempt, since its span is then arbitrary and the
        signs around it are cut by it. `unheard_is_miss` scores it as a miss of that sign instead and
        keeps the rest of the verdicts, which is what a story needs: it never stops, and a word left
        unsaid is a word left unsigned.

        `voice` is when the page heard the voice, its start and end in seconds of the audio; only the
        audio around it is aligned, since the rest is the wait before speaking and the silence after.

        A learner who signs in silence sends `cuts` instead of `audio`: a start and an end per sign,
        in seconds from the first frame, where the page saw the hands raised and lowered again."""
        if any(s not in index for s in sign):
            raise HTTPException(404, "unknown sign")
        values = decode(await landmarks.read(), await times.read(), handedness, width, height)
        if spoken and len(spoken) != len(sign):
            raise HTTPException(400, "a spoken word per sign, or none at all")
        if voice and len(voice) != 2:
            raise HTTPException(400, "a start and an end of the voice")
        if cuts:
            if len(cuts) != 2 * len(sign):
                raise HTTPException(400, "a start and an end per sign")
            edges = np.clip(np.round(np.array(cuts) * verifier.fps), 0, len(values)).astype(int)
            return {"threshold": threshold, "note": "", "signs": [judge(values[a:b], s, handedness, width, height) for a, b, s in zip(edges[::2], edges[1::2], sign)]}  # fmt: skip
        if audio is None:
            return {"threshold": threshold, "note": NOTES["no_audio"], "signs": []}
        words = list(spoken) or [spoken_word(s) for s in sign]
        # Timed in the log, since the alignment is nearly all of what an attempt costs.
        started = time.perf_counter()
        sound = decode_audio(await audio.read())
        decoded = time.perf_counter()
        if voice:
            sound, cut = trim(sound, voice)
            audio_offset -= cut  # the words are then timed from the trimmed audio's start
        spans = aligner(sound, words)
        print(f"an attempt of {len(values) / verifier.fps:.1f} s, {len(sound) / SAMPLE_RATE:.1f} s of audio: decoded in {decoded - started:.2f} s, aligned in {time.perf_counter() - decoded:.2f} s")  # fmt: skip
        if spans is None:
            return {"threshold": threshold, "note": NOTES["not_said"], "signs": []}
        # The alignment is a Viterbi path and always returns one, so silence aligns as readily as
        # speech; the score is what says the words were spoken at all (see takk.speech).
        unheard = [word for word, (_, _, score) in zip(words, spans) if score < MIN_WORD_SCORE]
        if unheard:
            print(f"an attempt was refused, the words scoring {[round(s, 2) for *_, s in spans]}")  # while MIN_WORD_SCORE is provisional
        if unheard and not unheard_is_miss:
            return {"threshold": threshold, "note": NOTES["not_heard"].format(words=" och ".join(unheard)), "signs": []}  # fmt: skip
        parts = split_speech(spans, audio_offset, len(values), verifier.fps)
        if len(parts) != len(sign) or any(part.stop - part.start < 2 for part in parts):
            return {"threshold": threshold, "note": NOTES["not_said"], "signs": []}
        signs = [judge(values[part], s, handedness, width, height) for part, s in zip(parts, sign)]
        for judged, word, (_, _, score) in zip(signs, words, spans):
            if score < MIN_WORD_SCORE:  # only with unheard_is_miss: the word was not said, so its sign was not made
                judged.update(correct=False, note=NOTES["not_heard"].format(words=word))
        return {"threshold": threshold, "note": "", "signs": signs}

    return app
