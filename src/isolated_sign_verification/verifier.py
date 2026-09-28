"""The verification model as an application uses it: whether a recording of a sign can be judged, and
how much it looks like each sign of a set of references.

An application needs this module and the landmark format (`sign_data.landmarks`), nothing else of
this package. How a model turns a recording into scores - the preparation, the mirroring to the
dominant hand, the embedding and the similarity - stays behind it, and so does what a score of a
reference set is made of. Nothing here is specific to a sign language: the references are any
prepared set of clips, so one model serves the glossary of any lexicon.

A verifier is saved as a directory of its own (`save`, `load`): the weights, the configs they are
built and fed by, the threshold, and the run and commit they came from. A reference set is a file of
its own (`References.save`), since one verifier scores against many.
"""

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from isolated_sign_verification.checks import Check, check
from isolated_sign_verification.dataset import SignDataset, clip_item, collate
from isolated_sign_verification.models import build_model
from isolated_sign_verification.preparation import ONE_HANDED, PrepConfig, PreparedData, hand_presence, hide_low_hands, mirror, prepare_clip
from sign_data.landmarks import VideoInfo

MODEL_FILE = "model.pt"
META_FILE = "verifier.json"


class Attempt(NamedTuple):
    """A recording of one sign: its landmarks, float32 (n_frames, N_LANDMARKS, 3), NaN where not
    detected, at the verifier's `fps`, from frames of `width` x `height` pixels, and the signer's
    dominant hand, "left" or "right"."""

    landmarks: np.ndarray
    width: int
    height: int
    handedness: str


@dataclass(frozen=True)
class References:
    """The signs an attempt is scored against, each from its reference clips. What a sign's row holds
    is the verifier's business; an application only keeps, saves and passes it back."""

    signs: list[str]
    means: np.ndarray  # (n_signs, dim), row i is signs[i]: the mean of its clips' unit-length embeddings

    def save(self, path: Path) -> None:
        pl.DataFrame(
            {"sign": self.signs, "mean": self.means.astype(np.float32)},
            schema={"sign": pl.String, "mean": pl.Array(pl.Float32, self.means.shape[1])},
        ).write_parquet(path)

    @classmethod
    def load(cls, path: Path) -> "References":
        table = pl.read_parquet(path)
        return cls(table["sign"].to_list(), table["mean"].to_numpy())


def sign_means(embeddings: np.ndarray, labels: np.ndarray, n_signs: int) -> np.ndarray:
    """The mean of each sign's unit-length clip embeddings, shape (n_signs, dim). Its dot product with
    a unit-length embedding is the mean cosine similarity to the sign's clips, as `evaluation` scores
    a sign from k references."""
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sums = np.zeros((n_signs, unit.shape[1]))
    np.add.at(sums, labels, unit)
    return sums / np.bincount(labels, minlength=n_signs)[:, None]


@torch.no_grad()
def embed(model: nn.Module, dataset, device: str, batch_size: int = 512, num_workers: int = 4, progress: bool = False) -> np.ndarray:
    """Embeddings of all clips of a (non-augmented) dataset, in order, with a progress bar if `progress`."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    parts = []
    for batch in tqdm(loader, desc="embedding", unit="batch", disable=not progress):
        with torch.autocast(device, dtype=torch.bfloat16):
            parts.append(model(batch["frames"].to(device), batch["hands"].to(device), batch["mask"].to(device)).float().cpu())
    return torch.cat(parts).numpy()


def _prep_config(values: dict) -> PrepConfig:
    return PrepConfig(**values | {"groups": tuple(values["groups"])})


def _commit() -> str | None:
    """The commit the code is at, so that a saved verifier can be traced to the code that reads it."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.stdout.strip() or None


class Verifier:
    """A trained model, the preparation its input goes through, and the score that counts as the sign."""

    def __init__(self, model: nn.Module, config: PrepConfig, threshold: float, meta: dict, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.config = config
        self.threshold = threshold
        self.meta = meta  # where it came from, and the training config the model is built from
        self.device = device

    @property
    def fps(self) -> float:
        """The frame rate an attempt is recorded at."""
        return self.config.fps

    @property
    def max_seconds(self) -> float:
        """The longest recording of one sign that is looked at in full."""
        return self.config.max_frames / self.config.fps

    @classmethod
    def from_run(cls, run_dir: Path, prepared: Path, threshold: float, device: str = "cpu") -> "Verifier":
        """The best model of a training run, fed clips prepared as the prepared set `prepared` is (the
        references it will score against)."""
        config = _prep_config(json.loads((prepared / "config.json").read_text()))
        train_config = json.loads((run_dir / "config.json").read_text())
        model = build_model(SimpleNamespace(**train_config), config)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu")["model"])
        return cls(model, config, threshold, {"run": run_dir.name, "commit": _commit(), "train_config": train_config}, device)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.to("cpu").state_dict()}, path / MODEL_FILE)
        self.model.to(self.device)
        meta = self.meta | {
            "created": date.today().isoformat(),
            "threshold": self.threshold,
            "model_sha256": hashlib.sha256((path / MODEL_FILE).read_bytes()).hexdigest(),
            "prep_config": dataclasses.asdict(self.config),
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, device: str = "cpu", threshold: float | None = None) -> "Verifier":
        """A saved verifier. `threshold` overrides the one it was saved with."""
        meta = json.loads((path / META_FILE).read_text())
        config = _prep_config(meta["prep_config"])
        model = build_model(SimpleNamespace(**meta["train_config"]), config)
        model.load_state_dict(torch.load(path / MODEL_FILE, map_location="cpu")["model"])
        return cls(model, config, meta["threshold"] if threshold is None else threshold, meta, device)

    def references(self, prepared: Path) -> References:
        """The signs of a prepared evaluation set (a `prepare_*` script's output, prepared with this
        verifier's config), each from all of its clips."""
        data = PreparedData(prepared)
        if data.config != self.config:
            raise ValueError(f"{prepared} is prepared with another config than the verifier's")
        clips = SignDataset(data, "test")
        embeddings = embed(self.model, clips, self.device, progress=True)
        return References(clips.signs, sign_means(embeddings, clips.labels, len(clips.signs)))

    def check(self, attempt: Attempt) -> Check:
        """Whether an attempt can be scored at all, and if not, why."""
        return check(attempt.landmarks, VideoInfo(self.fps, attempt.width, attempt.height), self.config)

    def scores(self, attempt: Attempt, references: References) -> np.ndarray:
        """A usable attempt's score for each sign of `references`, in the order of its `signs`; it
        counts as a sign when the score reaches `threshold`."""
        frames = self._prepare(attempt)
        if frames is None:
            raise ValueError("the attempt is not usable, see check")
        return references.means @ self._embed(frames)

    def _prepare(self, attempt: Attempt) -> np.ndarray | None:
        """Prepare an attempt as `prepare_store` prepares a clip, mirrored when its dominant hand is the
        left: a one-handed attempt's detected hand, a two-handed one's stated `handedness` (as
        `add_dominant_hands`, with the signer's handedness stated rather than inferred)."""
        aspect = attempt.width / attempt.height
        frames = prepare_clip(attempt.landmarks, self.fps, aspect, self.config)
        if frames is None:
            return None
        present = hand_presence(hide_low_hands(attempt.landmarks, aspect, self.config.max_hand_y))
        left, right, both = present[:, 0].sum(), present[:, 1].sum(), present.all(axis=1).sum()
        if both < ONE_HANDED * (left + right - both):
            dominant = "left" if left > right else "right"
        else:
            dominant = attempt.handedness
        return mirror(frames, self.config) if dominant == "left" else frames

    @torch.inference_mode()
    def _embed(self, frames: np.ndarray) -> np.ndarray:
        """The unit-length embedding of one prepared clip, as `embed` embeds the references."""
        batch = collate([clip_item(frames, [self.config.group_slices[hand] for hand in ("left_hand", "right_hand")])])
        with torch.autocast(self.device, dtype=torch.bfloat16):
            embedding = self.model(batch["frames"].to(self.device), batch["hands"].to(self.device), batch["mask"].to(self.device))
        embedding = embedding.float().cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)
