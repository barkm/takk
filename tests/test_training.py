import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import TrainConfig, phonology_targets, train, twin_matrix


def test_phonology_targets():
    clips = pl.DataFrame({
        "sign": ["APPLE", "APPLE", "TREE", "auslan:WHALE", "NIGHT"],
        "phonology.Handshape": ["x", "x", "open_b", None, "x"],
        "phonology.Contact": ["1", "1", "0", None, "0"],
    })  # fmt: skip

    targets, n_classes = phonology_targets(clips, ["APPLE", "NIGHT", "TREE", "auslan:WHALE"])

    assert n_classes == [2, 2]
    assert targets.tolist() == [[1, 1], [1, 0], [0, 0], [-1, -1]]  # classes in sorted order, -1 unknown


def test_twin_matrix():
    matrix = twin_matrix({("A", "C"), ("B", "HELD_OUT")}, ["A", "B", "C"])

    assert matrix.tolist() == [[False, False, True], [False, False, False], [True, False, False]]
    assert twin_matrix({("B", "HELD_OUT")}, ["A", "B"]) is None  # no pair among the training signs


def tiny_prepared(directory: Path, n_signs: int = 6, n_signers: int = 4, n_frames: int = 12) -> PreparedData:
    """A prepared directory of random clips: train signs S0.. and val signs V0.., one clip per signer."""
    config = PrepConfig()
    rng = np.random.default_rng(0)
    rows = [
        {"clip_id": f"{prefix}{sign}-{signer}", "sign": f"{prefix}{sign}", "signer": f"P{signer}", "split": split}
        for prefix, split in (("S", "train"), ("V", "val"))
        for sign in range(n_signs)
        for signer in range(n_signers)
    ]
    frames = rng.normal(size=(len(rows) * n_frames, len(config.landmarks), config.n_coords)).astype(np.float32)
    clips = pl.DataFrame(rows).with_columns(offset=pl.int_range(pl.len()) * n_frames, n_frames=pl.lit(n_frames))
    directory.mkdir()
    np.save(directory / "frames.npy", frames)
    clips.write_parquet(directory / "clips.parquet")
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    return PreparedData(directory)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training autocasts to bfloat16 on CUDA")
@pytest.mark.parametrize("settings", [{}, {"ema_decay": 0.9}])
def test_train_runs_and_writes_the_validation(tmp_path, settings):
    config = TrainConfig(hidden=16, embedding_dim=8, epochs=2, batch_size=8, num_workers=1, eval_every=1, **settings)
    train(config, tiny_prepared(tmp_path / "prepared"), tmp_path / "run", device="cuda")
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "val_summary.parquet").exists()
