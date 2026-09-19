"""Practicing signs: pick a sign of a glossary, sign it in front of the webcam, and learn whether it was that sign.

A small web app (`scripts/practice.py`, `web/practice.html`). The browser extracts the landmarks
itself, with the same HolisticLandmarker setup as `extraction.py` (see the browser extraction
findings in ROADMAP.md), and sends only those: the video never leaves the user's device. The backend
checks and prepares the attempt like any recording, embeds it, and scores it against the glossary
clips of the chosen sign: the mean cosine similarity to them, as `evaluation` scores a sign from k
references. The attempt counts as the sign when the score reaches a global threshold.
"""

import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from torch import nn

from isolated_sign_validation.collection import WEB_DIR, check_clip
from isolated_sign_validation.dataset import collate
from isolated_sign_validation.extraction import MODEL_PATH, VideoInfo
from isolated_sign_validation.landmarks import N_LANDMARKS
from isolated_sign_validation.preparation import ONE_HANDED, PrepConfig, hand_presence, hide_low_hands, mirror, prepare_clip


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
) -> FastAPI:
    """The practice app: the page, the glossary's signs with their clips (`references`, sign ->
    clip ids, in the order of the rows of `means`, see sign_means), the clips' videos (by clip id, see
    video_paths), the extraction model for the browser, and the scoring of attempts."""
    app = FastAPI()
    index = {sign: i for i, sign in enumerate(references)}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "practice.html")

    @app.get("/api/signs")
    def signs() -> dict:
        return {
            "signs": [{"sign": sign, "references": clips} for sign, clips in references.items()],
            "fps": config.fps,
            "max_seconds": config.max_frames / config.fps,
        }

    @app.get("/api/reference/{clip_id}")
    def reference(clip_id: str) -> FileResponse:
        if clip_id not in reference_videos:
            raise HTTPException(404, "unknown reference clip")
        return FileResponse(reference_videos[clip_id])

    @app.get("/api/model")
    def extraction_model() -> FileResponse:
        return FileResponse(MODEL_PATH)

    @app.post("/api/attempt")
    async def attempt(landmarks: UploadFile, sign: str = Form(), handedness: str = Form(), width: int = Form(), height: int = Form()) -> dict:  # fmt: skip
        """Score an attempt: its landmarks as float32 (n_frames, N_LANDMARKS, 3), NaN where not
        detected, at the preparation's frame rate, from frames of `width` x `height` pixels."""
        if sign not in index:
            raise HTTPException(404, "unknown sign")
        values = np.frombuffer(await landmarks.read(), dtype=np.float32)
        if handedness not in ("left", "right") or width <= 0 or height <= 0 or values.size % (N_LANDMARKS * 3):
            raise HTTPException(400, "malformed attempt")
        values = values.reshape(-1, N_LANDMARKS, 3)
        usable, note, _ = check_clip(values, VideoInfo(config.fps, width, height), config)
        frames = prepare_attempt(values, config.fps, width / height, handedness, config) if usable else None
        if frames is None:
            return {"usable": False, "note": note}
        score = float(means[index[sign]] @ embed_clip(model, frames, config, device))
        return {"usable": True, "note": note, "score": score, "threshold": threshold, "correct": score >= threshold}

    return app
