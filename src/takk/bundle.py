"""The glossary the app serves, as one directory that carries no training data.

A bundle is what `scripts/build_serving.py` writes from a training run and a prepared glossary, and
the only thing the API reads at startup (step 12 of ROADMAP.md, step 18 of ROADMAP-takk.md). It holds
the encoder's weights, one mean embedding per sign, the addresses its clips are watched at, and the
words a learner can search for — about 40 MB, against the 641 MB of prepared frames that loading a
`PreparedData` reads for nothing once the clip embeddings are cached.

Reading one imports `models.py` and `PrepConfig` and nothing else of the pipeline: not `dataset.py`,
not `PreparedData`, not `training.py`, which is what lets the deployed image leave the data packages
out. The mean embeddings sit in a column of `signs.parquet` beside the sign they belong to rather
than in a bare matrix, so a sign that the training never saw is added by appending a row.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from torch import nn

from isolated_sign_validation.models import build_model
from isolated_sign_validation.preparation import PrepConfig
from takk.vocabulary import Index

MODEL_FILE = "model.pt"
SIGNS_FILE = "signs.parquet"
VOCABULARY_FILE = "vocabulary.json"
META_FILE = "meta.json"


@dataclass(frozen=True)
class Bundle:
    """Everything the API serves: the signs in the order of the rows of `means`, each sign's clip
    addresses, the words that lead to them, the lexicon's description of each entry's form, the
    preparation an attempt goes through, the model that embeds it and what a score has to reach."""

    signs: list[str]
    means: np.ndarray  # (n_signs, dim); row i is signs[i]
    clips: dict[str, list[str]]
    vocabulary: Index
    forms: dict[str, str]
    config: PrepConfig
    threshold: float
    model: nn.Module
    meta: dict


def write(
    path: Path,
    signs: list[str],
    means: np.ndarray,
    clips: dict[str, list[str]],
    vocabulary: Index,
    forms: dict[str, str],
    config: PrepConfig,
    train_config: dict,
    threshold: float,
    state_dict: dict,
    source: dict,
) -> None:
    """Write a bundle. `source` names where it came from (the run and the glossary), which is kept in
    `meta.json` so a served model can always be traced back to what made it."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, path / MODEL_FILE)
    pl.DataFrame(
        {"sign": signs, "mean": means.astype(np.float32), "clips": [clips.get(sign, []) for sign in signs]},
        schema={"sign": pl.String, "mean": pl.Array(pl.Float32, means.shape[1]), "clips": pl.List(pl.String)},
    ).write_parquet(path / SIGNS_FILE)
    (path / VOCABULARY_FILE).write_text(
        json.dumps({"words": vocabulary.words, "themes": vocabulary.themes, "forms": forms}, ensure_ascii=False)
    )
    (path / META_FILE).write_text(
        json.dumps(
            {
                **source,
                "created": date.today().isoformat(),
                "threshold": threshold,
                "n_signs": len(signs),
                "dim": int(means.shape[1]),
                "model_sha256": hashlib.sha256((path / MODEL_FILE).read_bytes()).hexdigest(),
                "prep_config": dataclasses.asdict(config),
                "train_config": train_config,
            },
            indent=2,
        )
    )


def load(path: Path, threshold: float | None = None) -> Bundle:
    """Read a bundle, with the model on the CPU. `threshold` overrides the one it was built with."""
    meta = json.loads((path / META_FILE).read_text())
    config = PrepConfig(**meta["prep_config"] | {"groups": tuple(meta["prep_config"]["groups"])})
    model = build_model(SimpleNamespace(**meta["train_config"]), config)
    model.load_state_dict(torch.load(path / MODEL_FILE, map_location="cpu")["model"])
    model.eval()
    table = pl.read_parquet(path / SIGNS_FILE)
    vocabulary = json.loads((path / VOCABULARY_FILE).read_text())
    words = vocabulary["words"]
    by_sign: dict[str, dict] = {}
    for word in words:  # the words are in order, so the first one a sign is met under is its best
        by_sign.setdefault(word["sign"], word)
    return Bundle(
        signs=table["sign"].to_list(),
        means=table["mean"].to_numpy(),
        clips=dict(zip(table["sign"], table["clips"].to_list())),
        vocabulary=Index(words, vocabulary["themes"], by_sign),
        forms=vocabulary["forms"],
        config=config,
        threshold=meta["threshold"] if threshold is None else threshold,
        model=model,
        meta=meta,
    )
