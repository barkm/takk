from types import SimpleNamespace

import numpy as np

from isolated_sign_validation.models import build_model
from isolated_sign_validation.preparation import PrepConfig
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


def test_a_bundle_round_trips_what_the_api_serves(tmp_path):
    config = PrepConfig()
    model = build_model(SimpleNamespace(**TRAIN_CONFIG), config)
    means = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])
    words = [{"sign": "sts:hej-1", "id": "1"}, {"sign": "sts:mamma-2", "id": "2", "word": "mamma"}]
    vocabulary = Index(words, {"Hälsning": words[:1]}, {word["sign"]: word for word in words})

    bundle.write(
        tmp_path / "b",
        signs=["sts:hej-1", "sts:mamma-2"],
        means=means,
        clips={"sts:hej-1": ["https://example.test/hej.mp4"]},
        vocabulary=vocabulary,
        forms={"1": "Flata handen"},
        config=config,
        train_config=TRAIN_CONFIG,
        threshold=0.38,
        state_dict=model.state_dict(),
        source={"run": "a_run", "glossary": "a_glossary"},
    )
    read = bundle.load(tmp_path / "b")

    assert read.signs == ["sts:hej-1", "sts:mamma-2"]
    assert np.allclose(read.means, means)  # float32 in the bundle, float64 from sign_means
    assert read.clips == {"sts:hej-1": ["https://example.test/hej.mp4"], "sts:mamma-2": []}
    assert read.vocabulary == vocabulary and read.forms == {"1": "Flata handen"}
    assert read.config == config and read.threshold == 0.38
    assert read.meta["run"] == "a_run" and read.meta["glossary"] == "a_glossary" and read.meta["n_signs"] == 2
    # the model is rebuilt from what was stored beside the weights, without the training package
    assert [(key, tuple(value.shape)) for key, value in read.model.state_dict().items()] == [
        (key, tuple(value.shape)) for key, value in model.state_dict().items()
    ]


def test_a_threshold_can_be_overridden_when_it_is_read(tmp_path):
    config = PrepConfig()
    bundle.write(
        tmp_path / "b",
        signs=["a"],
        means=np.ones((1, 4)),
        clips={},
        vocabulary=Index([], {}, {}),
        forms={},
        config=config,
        train_config=TRAIN_CONFIG,
        threshold=0.38,
        state_dict=build_model(SimpleNamespace(**TRAIN_CONFIG), config).state_dict(),
        source={"run": "a_run", "glossary": "a_glossary"},
    )

    assert bundle.load(tmp_path / "b", threshold=0.5).threshold == 0.5
