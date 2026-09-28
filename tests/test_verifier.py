import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from isolated_sign_verification.models import build_model
from isolated_sign_verification.preparation import PrepConfig, mirror, prepare_clip
from isolated_sign_verification.verifier import Attempt, References, Verifier, sign_means
from sign_data.landmarks import LANDMARK_SLICES, N_LANDMARKS

CONFIG = PrepConfig()
POSE = LANDMARK_SLICES["pose"].start
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


def attempt_landmarks(hands: tuple[str, ...]) -> np.ndarray:
    """2 s of landmarks with shoulders in view and `hands` raised in the middle second."""
    x = np.full((60, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    x[:, LANDMARK_SLICES["face"]] = [0.5, 0.3, 0.0]
    x[:, LANDMARK_SLICES["pose"]] = [0.5, 0.5, 0.0]
    x[:, POSE + 11, :2] = [0.6, 0.5]
    x[:, POSE + 12, :2] = [0.4, 0.5]
    for hand in hands:
        x[15:45, LANDMARK_SLICES[hand]] = [0.35 if hand == "right_hand" else 0.65, 0.6, 0.0]
    return x


class Fixed(nn.Module):
    """A stand-in for the embedding model: the same embedding for every clip."""

    def forward(self, frames, hands, mask):
        return torch.tensor([[3.0, 4.0]]).expand(len(frames), 2)


def test_sign_means_give_the_mean_cosine_similarity():
    embeddings = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    means = sign_means(embeddings, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(means @ [1.0, 0.0], [0.5, np.sqrt(0.5)])


@pytest.mark.parametrize(
    "hands, handedness, mirrored",
    [
        (("right_hand",), "left", False),  # one-handed: the detected hand decides
        (("left_hand",), "right", True),
        (("left_hand", "right_hand"), "right", False),  # two-handed: the stated handedness decides
        (("left_hand", "right_hand"), "left", True),
    ],
)
def test_an_attempt_is_mirrored_when_its_dominant_hand_is_the_left(hands, handedness, mirrored):
    landmarks = attempt_landmarks(hands)
    frames = prepare_clip(landmarks, 30.0, 1.0, CONFIG)
    expected = mirror(frames, CONFIG) if mirrored else frames
    verifier = Verifier(Fixed(), CONFIG, 0.5, {})
    np.testing.assert_array_equal(verifier._prepare(Attempt(landmarks, 480, 480, handedness)), expected)


def test_an_attempt_is_checked_and_scored_against_every_reference():
    verifier = Verifier(Fixed(), CONFIG, 0.5, {})
    references = References(["A", "B"], np.array([[0.6, 0.8], [1.0, 0.0]]))  # cosine 1 and 0.6 with [3, 4]
    attempt = Attempt(attempt_landmarks(("right_hand",)), 640, 480, "right")

    assert verifier.check(attempt).usable
    np.testing.assert_allclose(verifier.scores(attempt, references), [1.0, 0.6], atol=1e-6)
    # the reason is a key an app words in its own language, with the values its wording takes
    empty = Attempt(np.full((60, N_LANDMARKS, 3), np.nan, dtype=np.float32), 640, 480, "right")
    assert verifier.check(empty)[:3] == (False, "no_hands", {})
    with pytest.raises(ValueError):
        verifier.scores(empty, references)


def test_a_saved_verifier_and_references_round_trip(tmp_path):
    model = build_model(SimpleNamespace(**TRAIN_CONFIG), CONFIG)
    Verifier(model, CONFIG, 0.38, {"run": "a_run", "train_config": TRAIN_CONFIG}).save(tmp_path / "v")
    References(["a", "b"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])).save(tmp_path / "r.parquet")

    read = Verifier.load(tmp_path / "v")
    assert read.config == CONFIG and read.threshold == 0.38 and read.fps == CONFIG.fps
    assert read.meta["run"] == "a_run" and "model_sha256" in json.loads((tmp_path / "v" / "verifier.json").read_text())
    # the model is rebuilt from what was stored beside the weights, without the training package
    assert [(key, tuple(value.shape)) for key, value in read.model.state_dict().items()] == [
        (key, tuple(value.shape)) for key, value in model.state_dict().items()
    ]
    assert Verifier.load(tmp_path / "v", threshold=0.5).threshold == 0.5

    references = References.load(tmp_path / "r.parquet")
    assert references.signs == ["a", "b"] and np.allclose(references.means, [[1, 0, 0, 0], [0, 0.5, 0.5, 0]])
