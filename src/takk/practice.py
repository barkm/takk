"""Practicing signs: pick a sign of a glossary, sign it in front of the webcam, and learn whether it was that sign.

A small web app (`main.py`, `web/practice.html`). The browser extracts the landmarks
itself, with the same HolisticLandmarker setup as `extraction.py` (see the browser extraction
findings in ROADMAP.md), and sends only those: the video never leaves the user's device. The backend
checks and prepares the attempt like any recording, embeds it, and scores it against the glossary
clips of the chosen sign: the mean cosine similarity to them, as `evaluation` scores a sign from k
references. The attempt counts as the sign when the score reaches a global threshold. The sign of
the whole glossary with the highest score is reported too, so a wrong attempt shows what it resembled.

An attempt can also be a sentence of several signs, as TAKK signs the key words of a spoken
sentence. The recording is split into its signs and each part is scored as an attempt of the sign at
its place in the sentence. With the sentence spoken aloud, the split follows the speech (`speech`):
the signer speaks a whole Swedish sentence and the key words are timed in it. Without audio it
follows the rests instead (split_signs), and the signer has to lower their hands between the signs.
"""

import re
from pathlib import Path

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

WEB_DIR = Path(__file__).parent / "web"

# Seconds without a raised hand that end a sign of a sentence. Inside a sign the hands are lost for
# at most 0.23 s in the Swedish browser recordings; lowering the hands and raising them again takes longer.
MIN_REST = 0.4


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


def split_signs(landmarks: np.ndarray, aspect: float, config: PrepConfig) -> list[slice]:
    """The signs of a recording at the preparation's frame rate: the runs of frames with a raised hand
    (resting hands count as not detected, see hide_low_hands) at least `MIN_REST` apart, without runs
    too short for a sign. Each slice reaches halfway into the rests around it."""
    frames = np.flatnonzero(hand_presence(hide_low_hands(landmarks, aspect, config.max_hand_y)).any(axis=1))
    if not len(frames):
        return []
    breaks = np.flatnonzero(np.diff(frames) > MIN_REST * config.fps)
    runs = [(a, b + 1) for a, b in zip(frames[np.r_[0, breaks + 1]], frames[np.r_[breaks, len(frames) - 1]])]
    runs = [(a, b) for a, b in runs if b - a >= config.min_hands * config.fps]
    if not runs:
        return []
    cuts = [0] + [(b + a) // 2 for (_, b), (a, _) in zip(runs[:-1], runs[1:])] + [len(landmarks)]
    return [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def spoken_word(sign: str) -> str:
    """The Swedish word a lexicon sign is signed for, which is what the signer says: its name without
    the source and the entry number ("sts:platta slag-25563" -> "platta slag")."""
    return re.sub(r"-\d+$", "", sign.removeprefix("sts:"))


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
    aligner: Aligner | None = None,
) -> FastAPI:
    """The practice app: the page, the glossary's signs with their clips (`references`, sign ->
    clip ids, in the order of the rows of `means`, see sign_means), the clips' videos (by clip id, see
    video_paths), the extraction model for the browser, and the scoring of attempts. With an
    `aligner` a spoken sentence is split into its signs by its words rather than by the rests."""
    app = FastAPI()
    names = list(references)
    index = {sign: i for i, sign in enumerate(names)}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "practice.html")

    @app.get("/api/signs")
    def signs() -> dict:
        return {
            "signs": [{"sign": sign, "references": clips} for sign, clips in references.items()],
            "fps": config.fps,
            "max_seconds": config.max_frames / config.fps,
            "edges": {group: edges.tolist() for group, edges in SKELETON_EDGES.items()},  # to draw the tracked landmarks
        }

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
        usable, note, _ = check_clip(values, VideoInfo(config.fps, width, height), config)
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
        frame) or, without it, by the rests; when the split fails, nothing is scored."""
        if any(s not in index for s in sign):
            raise HTTPException(404, "unknown sign")
        values = np.frombuffer(await landmarks.read(), dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or values.size % (N_LANDMARKS * 3):
            raise HTTPException(400, "malformed attempt")
        values = values.reshape(-1, N_LANDMARKS, 3)
        spans = aligner(decode_audio(await audio.read()), [spoken_word(s) for s in sign]) if aligner and audio and len(sign) > 1 else None  # fmt: skip
        if len(sign) == 1:
            parts, split = [slice(0, len(values))], "whole"
        elif spans is not None:
            parts, split = split_speech(spans, audio_offset, len(values), config.fps), "speech"
        else:
            parts, split = split_signs(values, width / height, config), "rests"
        if len(parts) != len(sign) or any(part.stop - part.start < 2 for part in parts):
            note = (
                "The words of the sentence were not found in what you said. Say each of them clearly."
                if split == "speech"
                else f"{len(parts)} signs were found, but the sentence has {len(sign)}. Lower your hands between the signs."
            )
            return {"threshold": threshold, "note": note, "split": split, "signs": []}
        signs = [judge(values[part], s, handedness, width, height) for part, s in zip(parts, sign)]
        return {"threshold": threshold, "note": "", "split": split, "signs": signs}

    return app
