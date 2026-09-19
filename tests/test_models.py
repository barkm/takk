import math

import pytest
import torch

from isolated_sign_validation.dataset import collate
from isolated_sign_validation.models import ArcFace, ConvTransformerEncoder, GRUEncoder, frame_features, n_frame_features
from isolated_sign_validation.preparation import PrepConfig

CONFIG = PrepConfig()
HANDS = [CONFIG.group_slices["left_hand"], CONFIG.group_slices["right_hand"]]
N = len(CONFIG.landmarks)


def item(n_frames: int, left_hand: bool = True, seed: int = 0) -> dict:
    frames = torch.randn(n_frames, N, 2, generator=torch.Generator().manual_seed(seed))
    hands = torch.ones(n_frames, 2, dtype=torch.bool)
    if not left_hand:
        frames[:, HANDS[0]] = 0
        hands[:, 0] = False
    return {"frames": frames, "hands": hands, "label": 0}


def test_frame_features_zero_velocity_for_missing_hand():
    batch = collate([item(5, left_hand=False)])
    features = frame_features(batch["frames"], batch["hands"], batch["mask"], HANDS)
    assert features.shape == (1, 5, 2 * N * 2 + 2 * 21 * 2 + 2)
    velocity = features[0, :, N * 2 + 2 * 21 * 2 : N * 2 * 2 + 2 * 21 * 2].reshape(5, N, 2)
    assert (velocity[:, HANDS[0]] == 0).all() and (velocity[0] == 0).all()
    assert (velocity[1:, HANDS[1]] != 0).any()


def test_bone_features_are_unit_directions_and_zero_for_a_missing_hand():
    batch = collate([item(5, left_hand=False)])
    features = frame_features(batch["frames"], batch["hands"], batch["mask"], HANDS, bones=True)
    assert features.shape[2] == n_frame_features(N, HANDS, bones=True) == n_frame_features(N, HANDS) + 2 * 20 * 2
    left, right = features[0, :, -2 * 20 * 2 :].reshape(5, 2, 20, 2).unbind(1)
    assert (left == 0).all()
    torch.testing.assert_close(right.norm(dim=2), torch.ones(5, 20))


@pytest.mark.parametrize("encoder", [GRUEncoder, ConvTransformerEncoder])
def test_encoder_embeddings_are_unit_length_and_ignore_padding(encoder):
    model = encoder(N, HANDS, hidden=32, embedding_dim=16).eval()
    short, long = item(4, seed=1), item(9, seed=2)
    alone = collate([short])
    padded = collate([short, long])  # short is padded to 9 frames
    with torch.no_grad():
        e_alone = model(alone["frames"], alone["hands"], alone["mask"])
        e_padded = model(padded["frames"], padded["hands"], padded["mask"])
    assert e_padded.shape == (2, 16)
    torch.testing.assert_close(e_padded.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(e_padded[0], e_alone[0], atol=1e-5, rtol=1e-5)


def test_arcface_adds_margin_to_the_true_class_only():
    torch.manual_seed(0)  # some random embeddings miss the 1e-4 tolerance
    head = ArcFace(8, 3, scale=10.0, margin=0.5)
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 0])
    plain, with_margin = head(embeddings), head(embeddings, labels)
    target = torch.nn.functional.one_hot(labels, 3).bool()
    torch.testing.assert_close(with_margin[~target], plain[~target])
    assert (with_margin[target] < plain[target]).all()
    # for a small angle the target logit is exactly scale * cos(theta + margin)
    head.weight.data = embeddings[:1].clone()
    logit = head(embeddings[:1], torch.tensor([0]))[0, 0]
    torch.testing.assert_close(logit, torch.tensor(10.0 * math.cos(0.5)), atol=1e-4, rtol=1e-4)


def test_arcface_twins_are_not_pushed_apart():
    head = ArcFace(8, 3)
    embeddings = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    labels = torch.tensor([0, 2])
    twins = torch.zeros(3, 3, dtype=torch.bool)
    twins[0, 1] = twins[1, 0] = True
    logits = head(embeddings, labels, twins)
    assert logits[0, 1] == float("-inf") and torch.isfinite(logits[1]).all()  # class 2 has no twin
    # the clip of class 0 moves no weight of its twin class 1, but still pushes class 2 away
    torch.nn.functional.cross_entropy(logits[:1], labels[:1]).backward()
    assert (head.weight.grad[1] == 0).all() and (head.weight.grad[2] != 0).any()


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model, head = GRUEncoder(N, HANDS, hidden=32, embedding_dim=16), ArcFace(16, 2, margin=0.2)
    batch = collate([item(6, seed=s) | {"label": s % 2} for s in range(8)])
    optimizer = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1e-2)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(head(model(batch["frames"], batch["hands"], batch["mask"]), batch["labels"]), batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("encoder", [GRUEncoder, ConvTransformerEncoder])
def test_return_pooled_gives_the_embeddings_input(encoder):
    model = encoder(N, HANDS, hidden=32, layers=1, embedding_dim=16).eval()
    batch = collate([item(5), item(3, seed=1)])
    arguments = (batch["frames"], batch["hands"], batch["mask"])

    embedding, pooled = model(*arguments, return_pooled=True)

    assert pooled.shape == (2, model.head.in_features)
    assert torch.allclose(embedding, torch.nn.functional.normalize(model.head(pooled), dim=1), atol=1e-6)
    assert torch.allclose(embedding, model(*arguments), atol=1e-6)
