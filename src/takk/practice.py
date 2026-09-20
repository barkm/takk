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
sentence. The recording is split into its signs by the speech and each part is scored as an attempt
of the sign at its place in the sentence: the signer speaks a whole Swedish sentence and the key
words are timed in it (`speech`). A sentence therefore needs the microphone. Splitting at the rests
between the signs instead was removed (see ROADMAP-takk.md step 3): it could not tell a sign the
signer skipped from a sign it had failed to find, so it voided the whole sentence and with it the
verdicts on the signs that were right, while the speech split scores a skipped sign as a miss.
"""

import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from torch import nn

from isolated_sign_validation.checks import check_clip
from isolated_sign_validation.dataset import collate
from isolated_sign_validation.extraction import MODEL_PATH, VideoInfo
from isolated_sign_validation.landmarks import N_LANDMARKS, SKELETON_EDGES
from isolated_sign_validation.preparation import ONE_HANDED, PrepConfig, hand_presence, hide_low_hands, mirror, prepare_clip
from takk.speech import Aligner, decode_audio, split_speech
from takk.vocabulary import spoken_word

# Everything the learner reads is Swedish (see ROADMAP-takk.md): the app is for practising TAKK.
NOTES = {
    "empty": "Inspelningen är tom.",
    "recorded": "Inspelat.",
    "no_hands": "Inga händer syntes. Har du händerna i bild när du tecknar?",
    "short": "Bara {seconds:.1f} s tecknande syntes. Spela in igen, lite långsammare.",
    "long": "{seconds:.0f} s tecknande syntes, vilket är för långt för ett tecken.",
    "no_body": "Din överkropp syntes inte. Sitt så att båda axlarna är i bild.",
    "lost_hand": "Går att bedöma, men en hand tappades i {lost:.0%} av bildrutorna medan du tecknade.",
    "ok": "Det ser bra ut.",
    "no_audio": "Mikrofonen behövs för en mening: orden du säger är det som delar upp inspelningen i tecken.",
    "not_said": "Meningens ord hittades inte i det du sa. Säg vart och ett av dem tydligt.",
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
    packs: list[dict] | None = None,
    forms: dict[str, str] | None = None,
) -> FastAPI:
    """The practice app: the page, the glossary's signs with their clips (`references`, sign ->
    clip ids, in the order of the rows of `means`, see sign_means), the clips' videos (by clip id, see
    video_paths), the extraction model for the browser, and the scoring of attempts. The `aligner`
    times a spoken sentence's words, which is what splits it into its signs."""
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

    @app.get("/api/packs")
    def practice_packs() -> dict:
        """The packs a learner can pick their daily practice from (`vocabulary.packs`): starter packs
        and the lexicon's categories alike, each a list of words with the sign that scores them."""
        return {"packs": packs or []}

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

    @app.post("/api/attempt")
    async def attempt(landmarks: UploadFile, sign: list[str] = Form(), handedness: str = Form(), width: int = Form(), height: int = Form(), audio: UploadFile | None = None, audio_offset: float = Form(0.0)) -> dict:  # fmt: skip
        """Score an attempt of one sign or a sentence of several (`sign` repeated, in order): its
        landmarks as float32 (n_frames, N_LANDMARKS, 3), NaN where not detected, at the preparation's
        frame rate, from frames of `width` x `height` pixels. A sentence is split into its signs
        first, by the sentence spoken in `audio` (recorded `audio_offset` seconds before the first
        frame), so it needs the microphone; when the split fails, nothing is scored."""
        if any(s not in index for s in sign):
            raise HTTPException(404, "unknown sign")
        values = np.frombuffer(await landmarks.read(), dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or values.size % (N_LANDMARKS * 3):
            raise HTTPException(400, "malformed attempt")
        values = values.reshape(-1, N_LANDMARKS, 3)
        if len(sign) == 1:  # one sign is the whole recording, so it needs no microphone
            parts, split = [slice(0, len(values))], "whole"
        elif audio is None:
            return {"threshold": threshold, "note": NOTES["no_audio"], "split": "speech", "signs": []}
        else:
            spans = aligner(decode_audio(await audio.read()), [spoken_word(s) for s in sign])
            parts, split = split_speech(spans, audio_offset, len(values), config.fps), "speech"
        if len(parts) != len(sign) or any(part.stop - part.start < 2 for part in parts):
            return {"threshold": threshold, "note": NOTES["not_said"], "split": split, "signs": []}
        signs = [judge(values[part], s, handedness, width, height) for part, s in zip(parts, sign)]
        return {"threshold": threshold, "note": "", "split": split, "signs": signs}

    return app
