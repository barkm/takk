"""Sign embedding models and the ArcFace classification head used to train them."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def frame_features(frames: torch.Tensor, hands: torch.Tensor, mask: torch.Tensor, hand_slices: list[slice]) -> torch.Tensor:
    """Per-frame input features from a batch (see dataset.collate): coordinates, each hand's shape
    relative to its wrist, velocities (zero where a landmark is missing in either frame), hand flags."""
    b, t, n_landmarks, _ = frames.shape
    present = torch.ones(b, t, n_landmarks, dtype=torch.bool, device=frames.device)
    for i, hand in enumerate(hand_slices):
        present[:, :, hand] = hands[:, :, i : i + 1]
    present &= mask[:, :, None]
    local = [frames[:, :, hand] - frames[:, :, hand.start : hand.start + 1] for hand in hand_slices]
    velocity = torch.zeros_like(frames)
    velocity[:, 1:] = (frames[:, 1:] - frames[:, :-1]) * (present[:, 1:] & present[:, :-1])[..., None]
    features = [frames.flatten(2), *(hand.flatten(2) for hand in local), velocity.flatten(2), hands.float()]
    return torch.cat(features, dim=2)


class GRUEncoder(nn.Module):
    """Bidirectional GRU over frame features, pooled over time into a unit-length embedding."""

    def __init__(self, n_landmarks: int, hand_slices: list[slice], hidden: int = 256, layers: int = 2, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.hand_slices = hand_slices
        n_features = 2 * n_landmarks * 2 + sum(2 * (s.stop - s.start) for s in hand_slices) + len(hand_slices)
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(4 * hidden, embedding_dim)  # mean and max pooling of both directions

    def forward(self, frames: torch.Tensor, hands: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input(frame_features(frames, hands, mask, self.hand_slices))
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        valid = mask[:, :, None]
        mean = (output * valid).sum(dim=1) / valid.sum(dim=1)
        maximum = output.masked_fill(~valid, -torch.inf).amax(dim=1)
        return F.normalize(self.head(torch.cat([mean, maximum], dim=1)), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin head (Deng et al., 2019): class logits from unit-length embeddings."""

    def __init__(self, embedding_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Logits; with `labels`, the true class gets the margin (for the training loss)."""
        cos = F.linear(embeddings, F.normalize(self.weight, dim=1)).float()
        if labels is None:
            return self.scale * cos
        sin = (1 - cos.square()).clamp(min=0).sqrt()
        with_margin = cos * math.cos(self.margin) - sin * math.sin(self.margin)  # cos(theta + margin)
        # beyond theta = pi - margin, cos(theta + margin) stops decreasing; use a linear penalty there instead
        with_margin = torch.where(cos > -math.cos(self.margin), with_margin, cos - self.margin * math.sin(self.margin))
        target = F.one_hot(labels, cos.shape[1]).bool()
        return self.scale * torch.where(target, with_margin, cos)
