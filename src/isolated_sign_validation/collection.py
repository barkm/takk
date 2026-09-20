"""Collecting self-recorded clips of known signs, for evaluation in the deployment setting.

A small web app (`scripts/collect.py`) shows a signer reference clips of a sign from a glossary, any
prepared evaluation set (ASL Citizen's test split, Svenskt teckenspråkslexikon, ...), and records
their attempt with their webcam. The recordings are a held-out evaluation set of the setting the
system is meant for: a user copying a dictionary clip, filmed with their own camera in their own
room. No score is ever shown, so the signer cannot retake until the model happens to agree, which
would bias the set toward clips the model already likes.

Recordings are stored as a raw dataset, ready for `datasets/recordings.py`:

- ``videos/<clip id>.mp4``: the recording, transcoded to a constant frame rate (the browser's
  MediaRecorder writes variable frame rate video, whose frame rate the extractor cannot read).
- ``clips.csv``: one row per recording, including the takes the signer discarded (`kept`), and the
  glossary clips the signer was shown (`references`).

Only the clip's validity is checked and reported back, by running the usual extraction and
preparation: whether it would survive `preparation.prepare_clip` at all, and in how many of its
frames a hand is detected. Otherwise a whole session can turn out to be unusable after the fact.
"""

import datetime as dt
import importlib
import random
import re
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import polars as pl
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from isolated_sign_validation.checks import check_clip
from isolated_sign_validation.extraction import extract_landmarks
from isolated_sign_validation.landmarks import SKELETON_EDGES
from isolated_sign_validation.preparation import PrepConfig

DATASET = "recordings"
RAW_DIR = Path("data/raw") / DATASET
WEB_DIR = Path("web")
NO_EVENT = "no_event"  # clips without signing, as in Slovo; negatives that look like real usage
# Prompts for the no_event clips, spread through a session. They have no reference clip to copy.
NO_EVENT_PROMPTS = [
    "Don't sign: just sit still and look at the camera.",
    "Don't sign: scratch your head, adjust your hair or your clothes.",
    "Don't sign: talk to the camera and gesture with your hands as you would while speaking.",
]
# One row per recording in clips.csv, including the takes the signer discarded.
SCHEMA = {
    "clip_id": pl.String,
    "session": pl.String,  # one signer's sitting, and the unit the take numbers count within
    "signer": pl.String,
    "handedness": pl.String,  # the hand the signer signs with, needed to mirror two-handed clips
    "sign": pl.String,
    "take": pl.Int64,
    "recorded_at": pl.String,
    "usable": pl.Boolean,  # whether preparation would keep the clip (see check_clip)
    "note": pl.String,
    "hand_share": pl.Float64,
    "kept": pl.Boolean,  # whether the signer kept this take rather than recording the sign again
    "confident": pl.Boolean,  # whether the signer felt sure of the sign, to separate fumbled attempts
    "references": pl.String,  # the glossary clips shown to copy, joined by ";"; null in older rows
}


def session_prompts(glossary: pl.DataFrame, n_signs: int, takes: int, n_references: int, seed: int) -> list[dict]:
    """What to record, in order: `takes` prompts for each of `n_signs` signs of `glossary`, plus no_event prompts.

    `glossary` holds the clips a recording is scored against (clip_id, sign and signer columns): the
    test split of a prepared evaluation set, so its signs are unseen by the model. Each prompt carries
    up to `n_references` of those clips by different signers (a dataset without signer ids counts as
    one signer), since copying a single clip makes the recording partly a mimicry of that one
    performance. The shown clips stay in the glossary, as copying the dictionary clip is what a user
    does; they are stored with every take (`references`), so an evaluation can also leave them out.
    """
    rng = random.Random(seed)
    signs = sorted(rng.sample(sorted(glossary["sign"].unique()), n_signs))
    by_signer: dict[str, dict[str | None, list[str]]] = {sign: {} for sign in signs}
    for row in glossary.filter(pl.col("sign").is_in(signs)).sort("clip_id").iter_rows(named=True):
        by_signer[row["sign"]].setdefault(row["signer"], []).append(row["clip_id"])
    prompts = []
    for sign in signs:
        signers = rng.sample(sorted(by_signer[sign], key=str), min(n_references, len(by_signer[sign])))
        references = [rng.choice(by_signer[sign][signer]) for signer in signers]
        prompts += [{"sign": sign, "instruction": None, "references": references}] * takes
    for i, instruction in enumerate(NO_EVENT_PROMPTS):
        position = max(1, round((i + 1) * len(prompts) / (len(NO_EVENT_PROMPTS) + 1)))
        prompts.insert(position + i, {"sign": NO_EVENT, "instruction": instruction, "references": []})
    return prompts


def video_paths(clips: pl.DataFrame) -> dict[str, str]:
    """The video file of each of `clips` (clip_id and dataset columns), from its dataset's adapter."""
    paths = {}
    for dataset in clips["dataset"].unique():
        adapter = importlib.import_module(f"isolated_sign_validation.datasets.{dataset}")
        videos = adapter.read_videos(adapter.RAW_DIR)
        paths |= dict(zip(videos["clip_id"], videos["path"]))
    return {clip_id: str(paths[clip_id]) for clip_id in clips["clip_id"]}


def read_clips(path: Path) -> pl.DataFrame:
    """A clips.csv; columns added to SCHEMA after the file was written are null."""
    clips = pl.read_csv(path, schema_overrides={column: dtype for column, dtype in SCHEMA.items()})
    missing = [pl.lit(None, dtype).alias(column) for column, dtype in SCHEMA.items() if column not in clips.columns]
    return clips.with_columns(missing).select(list(SCHEMA))


def transcode(source: Path, target: Path, fps: float) -> None:
    """Rewrite a recording at a constant frame rate, without audio.

    The browser records variable frame rate video, where the extractor's single frame rate per clip
    is meaningless; the frame rate drives trimming, gap interpolation and resampling in preparation.
    """
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-an", "-r", str(fps), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)


def overlay(landmarks: np.ndarray) -> dict:
    """The skeletons to draw over the playback of a recording, as JSON.

    `frames` holds x, y (fractions of the frame) of the drawn landmarks per frame, null where not
    detected; `edges` the lines of each skeleton, as pairs of indices into those landmarks.
    """
    points = np.unique(np.concatenate(list(SKELETON_EDGES.values())))
    xy = np.round(landmarks[:, points, :2].astype(float), 3).reshape(len(landmarks), -1)
    return {
        "frames": np.where(np.isnan(xy), None, xy).tolist(),
        "edges": {group: np.searchsorted(points, edges).tolist() for group, edges in SKELETON_EDGES.items()},
    }


class Recordings:
    """The stored recordings: the video files and the clip table, kept in sync on disk."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.videos = raw_dir / "videos"
        self.videos.mkdir(parents=True, exist_ok=True)
        self.path = raw_dir / "clips.csv"
        self.clips = read_clips(self.path) if self.path.exists() else pl.DataFrame(schema=SCHEMA)
        self.lock = threading.Lock()

    def _write(self) -> None:
        self.clips.write_csv(self.path)

    def take(self, session: str, sign: str) -> int:
        """The number this recording of `sign` is in `session`, counting discarded takes."""
        return self.clips.filter((pl.col("session") == session) & (pl.col("sign") == sign)).height + 1

    def add(self, row: dict) -> None:
        with self.lock:
            self.clips = pl.concat([self.clips, pl.DataFrame([row], schema=SCHEMA)])
            self._write()

    def keep(self, clip_id: str, confident: bool) -> None:
        """Mark a take as the one the signer kept for its prompt."""
        with self.lock:
            if clip_id not in self.clips["clip_id"]:
                raise KeyError(clip_id)
            keep = pl.col("clip_id") == clip_id
            self.clips = self.clips.with_columns(
                kept=pl.when(keep).then(True).otherwise(pl.col("kept")),
                confident=pl.when(keep).then(confident).otherwise(pl.col("confident")),
            )
            self._write()


def create_app(prompts: list[dict], recordings: Recordings, reference_videos: dict[str, str], config: PrepConfig) -> FastAPI:
    """The collection app: the page, the prompts with their reference clips (video files by clip id,
    see video_paths), and the uploads."""
    app = FastAPI()
    shown = {prompt["sign"]: prompt["references"] for prompt in prompts}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "collect.html")

    @app.get("/api/prompts")
    def get_prompts() -> dict:
        return {"prompts": prompts, "max_seconds": config.max_frames / config.fps}

    @app.get("/api/reference/{name}")
    def reference(name: str) -> FileResponse:
        if name not in reference_videos:
            raise HTTPException(404, "unknown reference clip")
        return FileResponse(reference_videos[name])

    @app.post("/api/recordings")
    async def add_recording(video: UploadFile, session: str = Form(), signer: str = Form(), handedness: str = Form(), sign: str = Form()) -> dict:  # fmt: skip
        take = recordings.take(session, sign)
        clip_id = f"{session}_{re.sub(r'[^A-Za-z0-9-]', '_', sign)}_{take}"
        upload = recordings.videos / f"{clip_id}.upload"
        with upload.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        path = recordings.videos / f"{clip_id}.mp4"
        try:
            transcode(upload, path, config.fps)
        finally:
            upload.unlink()
        landmarks, info = extract_landmarks(path)
        usable, note, hand_share = check_clip(landmarks, info, config, signing=sign != NO_EVENT)
        recordings.add(
            {
                "clip_id": clip_id, "session": session, "signer": signer, "handedness": handedness,
                "sign": sign, "take": take, "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "usable": usable, "note": note, "hand_share": round(hand_share, 3), "kept": False, "confident": False,
                "references": ";".join(shown.get(sign, [])),
            }
        )  # fmt: skip
        return {"clip_id": clip_id, "usable": usable, "note": note, "hand_share": hand_share, "fps": info.fps, "overlay": overlay(landmarks)}  # fmt: skip

    @app.get("/api/recordings/{clip_id}")
    def recording(clip_id: str) -> FileResponse:
        if clip_id not in recordings.clips["clip_id"]:
            raise HTTPException(404, "unknown clip")
        return FileResponse(recordings.videos / f"{clip_id}.mp4")

    @app.post("/api/keep")
    def keep(clip_id: str = Form(), confident: bool = Form()) -> dict:
        try:
            recordings.keep(clip_id, confident)
        except KeyError:
            raise HTTPException(404, "unknown clip")
        return {"kept": clip_id}

    return app
