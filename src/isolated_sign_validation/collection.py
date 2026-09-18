"""Collecting self-recorded clips of known signs, for evaluation in the deployment setting.

A small web app (`scripts/collect.py`) shows a signer reference clips of a sign from ASL Citizen, or
from Svenskt teckenspråkslexikon with `--lexicon`, and records their attempt with their webcam. The recordings are a held-out evaluation set of the setting
the system is meant for: a user copying a dictionary clip, filmed with their own camera in their own
room. No score is ever shown, so the signer cannot retake until the model happens to agree, which
would bias the set toward clips the model already likes.

Recordings are stored as a raw dataset, ready for `datasets/recordings.py`:

- ``videos/<clip id>.mp4``: the recording, transcoded to a constant frame rate (the browser's
  MediaRecorder writes variable frame rate video, whose frame rate the extractor cannot read).
- ``clips.csv``: one row per recording, including the takes the signer discarded (`kept`).

Only the clip's validity is checked and reported back, by running the usual extraction and
preparation: whether it would survive `preparation.prepare_clip` at all, and in how many of its
frames a hand is detected. Otherwise a whole session can turn out to be unusable after the fact.
"""

import datetime as dt
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

from isolated_sign_validation.datasets import asl_citizen, sts_lexikon
from isolated_sign_validation.extraction import extract_landmarks
from isolated_sign_validation.preparation import PrepConfig, hand_presence, hide_low_hands, prepare_clip
from isolated_sign_validation.splits import sign_split

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
}


def session_prompts(raw_dir: Path, n_signs: int, takes: int, n_references: int, seed: int) -> list[dict]:
    """What to record, in order: `takes` prompts for each of `n_signs` held-out signs, plus no_event prompts.

    Signs are drawn from the test split of the ASL Citizen glosses, so they are unseen by the model.
    Each prompt carries reference clips of the sign by `n_references` different signers: copying a
    single clip would make the recording a mimicry of that one performance.

    The reference clips are never clips of the official test split, which is the glossary a recording
    is scored against; a recording would otherwise be compared against the very clip it copied. Clips
    of a held-out sign by other signers belong to no split (`splits.assign_splits`), so they are
    neither trained on nor scored against.
    """
    videos = asl_citizen.read_videos(raw_dir)
    held_out = sorted(sign for sign in videos["Gloss"].unique() if sign_split(sign) == "test")
    rng = random.Random(seed)
    signs = sorted(rng.sample(held_out, n_signs))
    scored = pl.read_csv(raw_dir / "splits" / "test.csv")["Video file"]
    videos = videos.filter(~pl.col("Video file").is_in(scored))

    by_signer: dict[str, dict[str, list[str]]] = {sign: {} for sign in signs}
    for row in videos.filter(pl.col("Gloss").is_in(signs)).sort("Video file").iter_rows(named=True):
        by_signer[row["Gloss"]].setdefault(row["Participant ID"], []).append(row["Video file"])
    prompts = []
    for sign in signs:
        signers = rng.sample(sorted(by_signer[sign]), min(n_references, len(by_signer[sign])))
        references = [rng.choice(by_signer[sign][signer]) for signer in signers]
        prompts += [{"sign": sign, "instruction": None, "references": references}] * takes
    return with_no_event(prompts)


def lexicon_prompts(raw_dir: Path, n_signs: int, takes: int, seed: int) -> list[dict]:
    """What to record, in order: `takes` prompts for each of `n_signs` Swedish Sign Language signs from
    Svenskt teckenspråkslexikon, plus no_event prompts. The model has never seen the language.

    Only the lexicon's sign classes can be recorded, since each has several recordings of one sign
    form (`sts_lexikon.sign_classes`): the prompt shows one of them (`sts_lexikon.shown_clips`) and
    the recording is scored against the others, never against the clip it copied. An entry with a
    single clip would be both. So one reference clip is shown, not several signers' as for ASL
    Citizen: most classes have only two clips. References are paths below `raw_dir`.
    """
    videos = sts_lexikon.read_videos(raw_dir)
    shown = videos.filter(pl.col("clip_id").is_in(sts_lexikon.shown_clips(videos))).sort("sign")
    rng = random.Random(seed)
    prompts = []
    for row in sorted(rng.sample(shown.to_dicts(), n_signs), key=lambda row: row["sign"]):
        reference = str(Path(row["path"]).relative_to(raw_dir))
        prompts += [{"sign": sts_lexikon.LABEL_PREFIX + row["sign"], "instruction": None, "references": [reference]}] * takes
    return with_no_event(prompts)


def with_no_event(prompts: list[dict]) -> list[dict]:
    """`prompts` with the no_event prompts spread through them."""
    for i, instruction in enumerate(NO_EVENT_PROMPTS):
        position = max(1, round((i + 1) * len(prompts) / (len(NO_EVENT_PROMPTS) + 1)))
        prompts.insert(position + i, {"sign": NO_EVENT, "instruction": instruction, "references": []})
    return prompts


def transcode(source: Path, target: Path, fps: float) -> None:
    """Rewrite a recording at a constant frame rate, without audio.

    The browser records variable frame rate video, where the extractor's single frame rate per clip
    is meaningless; the frame rate drives trimming, gap interpolation and resampling in preparation.
    """
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-an", "-r", str(fps), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)


def check_clip(video: Path, config: PrepConfig) -> tuple[bool, str, float]:
    """Whether a recording can be used, why not, and how steadily a hand was detected while signing.

    Runs the extraction and preparation the clip would go through later, so that an unusable
    recording is caught while the signer can still redo it. The hand share is measured over the
    signing itself (the first to the last frame with a hand), not over the whole recording: the rest
    before and after the sign has no hands in view by design, so over the whole clip even a clean
    recording scores around 0.4.
    """
    landmarks, info = extract_landmarks(video)
    aspect = info.width / info.height if info.height else 1.0
    present = hand_presence(hide_low_hands(landmarks, aspect, config.max_hand_y)).any(axis=1)
    usable = prepare_clip(landmarks, info.fps, aspect, config) is not None
    with_hands = np.flatnonzero(present)
    seconds = (with_hands[-1] - with_hands[0] + 1) / info.fps if len(with_hands) else 0.0
    hand_share = float(present[with_hands[0] : with_hands[-1] + 1].mean()) if len(with_hands) else 0.0
    if not len(landmarks):
        return False, "The recording is empty.", 0.0
    if not len(with_hands):
        return False, "No hands were detected. Are your hands inside the frame while signing?", hand_share
    if seconds < config.min_hands:
        return False, f"Only {seconds:.1f} s of signing was detected. Record again, a little slower.", hand_share
    if seconds > config.max_hands:
        return False, f"{seconds:.0f} s of signing was detected, which is too long for one sign.", hand_share
    if not usable:
        return False, "Your upper body was not detected. Sit so that both shoulders are in the frame.", hand_share
    if hand_share < 0.8:
        return True, f"Usable, but a hand was lost in {1 - hand_share:.0%} of the frames while signing.", hand_share
    return True, "Looks good.", hand_share


class Recordings:
    """The stored recordings: the video files and the clip table, kept in sync on disk."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.videos = raw_dir / "videos"
        self.videos.mkdir(parents=True, exist_ok=True)
        self.path = raw_dir / "clips.csv"
        self.clips = pl.read_csv(self.path, schema=SCHEMA) if self.path.exists() else pl.DataFrame(schema=SCHEMA)
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


def create_app(prompts: list[dict], recordings: Recordings, reference_dir: Path, config: PrepConfig) -> FastAPI:
    """The collection app: the page, the prompts with their reference clips (paths below
    `reference_dir`), and the uploads."""
    app = FastAPI()
    reference_videos = {reference for prompt in prompts for reference in prompt["references"]}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "collect.html")

    @app.get("/api/prompts")
    def get_prompts() -> dict:
        return {"prompts": prompts, "max_seconds": config.max_frames / config.fps}

    @app.get("/api/reference/{name:path}")
    def reference(name: str) -> FileResponse:
        if name not in reference_videos:  # also keeps requests inside reference_dir
            raise HTTPException(404, "unknown reference clip")
        return FileResponse(reference_dir / name)

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
        usable, note, hand_share = check_clip(path, config)
        recordings.add(
            {
                "clip_id": clip_id, "session": session, "signer": signer, "handedness": handedness,
                "sign": sign, "take": take, "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "usable": usable, "note": note, "hand_share": round(hand_share, 3), "kept": False, "confident": False,
            }
        )  # fmt: skip
        return {"clip_id": clip_id, "usable": usable, "note": note, "hand_share": hand_share}

    @app.post("/api/keep")
    def keep(clip_id: str = Form(), confident: bool = Form()) -> dict:
        try:
            recordings.keep(clip_id, confident)
        except KeyError:
            raise HTTPException(404, "unknown clip")
        return {"kept": clip_id}

    return app
