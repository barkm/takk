"""Sign embedding models and the ArcFace classification head used to train them."""

import math

import torch
import torch.nn.functional as F
from torch import nn


# The 20 bones of a hand, from the wrist out along each finger (MediaPipe hand landmark order).
HAND_BONES = [(0 if joint == 0 else 4 * finger + joint, 4 * finger + joint + 1) for finger in range(5) for joint in range(4)]


def n_frame_features(n_landmarks: int, hand_slices: list[slice], n_coords: int = 2, bones: bool = False) -> int:
    hand_points = sum(s.stop - s.start for s in hand_slices) + (len(HAND_BONES) * len(hand_slices) if bones else 0)
    return n_coords * (2 * n_landmarks + hand_points) + len(hand_slices)


def frame_features(frames: torch.Tensor, hands: torch.Tensor, mask: torch.Tensor, hand_slices: list[slice], bones: bool = False) -> torch.Tensor:
    """Per-frame input features from a batch (see dataset.collate): coordinates, each hand's shape
    relative to its wrist, velocities (zero where a landmark is missing in either frame), hand flags;
    with `bones`, also each hand's bone directions (unit vectors, zero for a missing hand), a hand
    shape independent of the hand's size in the image."""
    b, t, n_landmarks, _ = frames.shape
    present = torch.ones(b, t, n_landmarks, dtype=torch.bool, device=frames.device)
    for i, hand in enumerate(hand_slices):
        present[:, :, hand] = hands[:, :, i : i + 1]
    present &= mask[:, :, None]
    local = [frames[:, :, hand] - frames[:, :, hand.start : hand.start + 1] for hand in hand_slices]
    velocity = torch.zeros_like(frames)
    velocity[:, 1:] = (frames[:, 1:] - frames[:, :-1]) * (present[:, 1:] & present[:, :-1])[..., None]
    features = [frames.flatten(2), *(hand.flatten(2) for hand in local), velocity.flatten(2), hands.float()]
    if bones:
        start, end = zip(*HAND_BONES)
        for hand in hand_slices:
            points = frames[:, :, hand]
            features.append(F.normalize(points[:, :, list(end)] - points[:, :, list(start)], dim=3).flatten(2))
    return torch.cat(features, dim=2)


class GRUEncoder(nn.Module):
    """Bidirectional GRU over frame features, pooled over time into a unit-length embedding."""

    def __init__(
        self, n_landmarks: int, hand_slices: list[slice], hidden: int = 256, layers: int = 2, embedding_dim: int = 256,
        dropout: float = 0.2, n_coords: int = 2, bones: bool = False,
    ):  # fmt: skip
        super().__init__()
        self.hand_slices, self.bones = hand_slices, bones
        n_features = n_frame_features(n_landmarks, hand_slices, n_coords, bones)
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(4 * hidden, embedding_dim)  # mean and max pooling of both directions

    def forward(self, frames: torch.Tensor, hands: torch.Tensor, mask: torch.Tensor, return_pooled: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The embedding, and with `return_pooled` also the pooled encoder output it is projected from."""
        x = self.input(frame_features(frames, hands, mask, self.hand_slices, self.bones))
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        valid = mask[:, :, None]
        mean = (output * valid).sum(dim=1) / valid.sum(dim=1)
        maximum = output.masked_fill(~valid, -torch.inf).amax(dim=1)
        pooled = torch.cat([mean, maximum], dim=1)
        embedding = F.normalize(self.head(pooled), dim=1)
        return (embedding, pooled) if return_pooled else embedding


class ConvBlock(nn.Module):
    """Residual block: depthwise temporal convolution, then a gated pointwise projection."""

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise = nn.Linear(dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).masked_fill(~mask[:, :, None], 0)  # padding must not leak into real frames
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.glu(self.pointwise(F.silu(y)), dim=2))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention over the real frames."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.dropout(self.attention(y, y, y, key_padding_mask=~mask, need_weights=False)[0])
        return x + self.dropout(self.feedforward(self.norm2(x)))


class ConvTransformerEncoder(nn.Module):
    """Stacks of three convolution blocks and one transformer block, pooled into a unit-length embedding."""

    def __init__(
        self, n_landmarks: int, hand_slices: list[slice], hidden: int = 192, layers: int = 2,
        embedding_dim: int = 256, dropout: float = 0.2, heads: int = 4, kernel_size: int = 17, n_coords: int = 2, bones: bool = False,
    ):  # fmt: skip
        super().__init__()
        self.hand_slices, self.bones = hand_slices, bones
        n_features = n_frame_features(n_landmarks, hand_slices, n_coords, bones)
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        blocks = []
        for _ in range(layers):
            blocks += [ConvBlock(hidden, kernel_size, dropout) for _ in range(3)] + [TransformerBlock(hidden, heads, dropout)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(2 * hidden, embedding_dim)  # mean and max pooling

    def forward(self, frames: torch.Tensor, hands: torch.Tensor, mask: torch.Tensor, return_pooled: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The embedding, and with `return_pooled` also the pooled encoder output it is projected from."""
        x = self.input(frame_features(frames, hands, mask, self.hand_slices, self.bones))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        valid = mask[:, :, None]
        mean = (x * valid).sum(dim=1) / valid.sum(dim=1)
        maximum = x.masked_fill(~valid, -torch.inf).amax(dim=1)
        pooled = torch.cat([mean, maximum], dim=1)
        embedding = F.normalize(self.head(pooled), dim=1)
        return (embedding, pooled) if return_pooled else embedding


class ArcFace(nn.Module):
    """Additive angular margin head (Deng et al., 2019): class logits from unit-length embeddings."""

    def __init__(self, embedding_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None, twins: torch.Tensor | None = None) -> torch.Tensor:
        """Logits; with `labels`, the true class gets the margin (for the training loss). `twins`, a
        boolean (n_classes, n_classes) matrix of classes known to be one sign form, sets each clip's
        twin classes to -inf, so the loss neither pushes the twins apart nor pulls them together."""
        cos = F.linear(embeddings, F.normalize(self.weight, dim=1)).float()
        if labels is None:
            return self.scale * cos
        sin = (1 - cos.square()).clamp(min=0).sqrt()
        with_margin = cos * math.cos(self.margin) - sin * math.sin(self.margin)  # cos(theta + margin)
        # beyond theta = pi - margin, cos(theta + margin) stops decreasing; use a linear penalty there instead
        with_margin = torch.where(cos > -math.cos(self.margin), with_margin, cos - self.margin * math.sin(self.margin))
        target = F.one_hot(labels, cos.shape[1]).bool()
        logits = self.scale * torch.where(target, with_margin, cos)
        return logits if twins is None else logits.masked_fill(twins[labels], float("-inf"))
