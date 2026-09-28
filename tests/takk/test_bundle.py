from types import SimpleNamespace

import numpy as np

from isolated_sign_verification.models import build_model
from isolated_sign_verification.preparation import PrepConfig
from isolated_sign_verification.verifier import References, Verifier
from takk import bundle
from takk.vocabulary import Index

TRAIN_CONFIG = {
    "encoder": "gru",
    "hidden": 8,
    "layers": 1,
    "heads": 2,
    "kernel_size": 3,
    "embedding_dim": 4,
    "dropout": 0.0,
    "hand_bones": False,
    "attention_pool": False,
}


def write(path, threshold: float = 0.38) -> Index:
    """A bundle of two signs at `path`, and the vocabulary written into it."""
    config = PrepConfig()
    verifier = Verifier(build_model(SimpleNamespace(**TRAIN_CONFIG), config), config, threshold, {"run": "a_run", "train_config": TRAIN_CONFIG})  # fmt: skip
    references = References(["sts:hej-1", "sts:mamma-2"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]))
    words = [{"sign": "sts:hej-1", "id": "1"}, {"sign": "sts:mamma-2", "id": "2", "word": "mamma"}]
    vocabulary = Index(words, {"Hälsning": words[:1]}, {word["sign"]: word for word in words})
    clips = {"sts:hej-1": ["https://example.test/hej.mp4"]}
    bundle.write(path, verifier, references, clips, vocabulary, {"1": "Flata handen"}, "a_glossary")
    return vocabulary


def test_a_bundle_round_trips_what_the_api_serves(tmp_path):
    vocabulary = write(tmp_path / "b")
    read = bundle.load(tmp_path / "b")

    assert read.references.signs == ["sts:hej-1", "sts:mamma-2"]
    assert read.clips == {"sts:hej-1": ["https://example.test/hej.mp4"], "sts:mamma-2": []}
    assert read.vocabulary == vocabulary and read.forms == {"1": "Flata handen"}
    assert read.verifier.threshold == 0.38 and read.verifier.meta["run"] == "a_run"
    assert read.meta["glossary"] == "a_glossary" and read.meta["n_signs"] == 2


def test_a_threshold_can_be_overridden_when_it_is_read(tmp_path):
    write(tmp_path / "b")

    assert bundle.load(tmp_path / "b", threshold=0.5).verifier.threshold == 0.5
