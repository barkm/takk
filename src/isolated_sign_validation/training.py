"""Training of sign embedding models with an ArcFace classification loss over the training signs.

Optionally, linear heads on the embedding also predict each training sign's phonological features
(prepared clip columns `phonology.<feature>`, e.g. from ASL-LEX), as an auxiliary loss, either from
the embedding itself or from the pooled encoder output it is projected from (`phonology_input`); clips
of signs without features only get the ArcFace loss. The heads are only used in training.

A run directory gets the config, per-epoch metrics, training curves, the checkpoint with the lowest
validation EER at k = 1 (validation AUC is close to saturated), and the full validation evaluation
of that checkpoint.
"""

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless machine: figures are saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from isolated_sign_validation.dataset import AugmentConfig, SignDataset, collate
from isolated_sign_validation.evaluation import cosine_similarity, evaluate
from isolated_sign_validation.models import ArcFace, ConvTransformerEncoder, GRUEncoder
from isolated_sign_validation.preparation import PreparedData


@dataclass(frozen=True)
class TrainConfig:
    encoder: str = "gru"  # or "conv_transformer"
    hidden: int = 256  # model width
    layers: int = 2  # GRU layers, or stacks of 3 convolution blocks and a transformer block
    heads: int = 4  # attention heads (conv_transformer)
    kernel_size: int = 17  # temporal convolution kernel (conv_transformer)
    embedding_dim: int = 256
    dropout: float = 0.4
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3
    phonology_weight: float = 0.0  # weight of the auxiliary phonological feature loss (0: off)
    phonology_input: str = "embedding"  # what the feature heads read: "embedding" or "pooled" (the encoder output)
    augment: AugmentConfig = dataclasses.field(default_factory=AugmentConfig)
    # training signers left out of training, to measure generalization to unseen signers
    holdout_signers: tuple[str, ...] = ()
    # share of the training signs to train on (a random subset, nested across fractions for a given seed)
    train_sign_fraction: float = 1.0
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    eval_every: int = 2  # epochs between validation checks
    num_workers: int = 8
    seed: int = 0


def build_model(config: TrainConfig, data: PreparedData) -> nn.Module:
    hands = [data.config.group_slices[hand] for hand in ("left_hand", "right_hand")]
    n_landmarks = len(data.config.landmarks)
    if config.encoder == "gru":
        return GRUEncoder(n_landmarks, hands, config.hidden, config.layers, config.embedding_dim, config.dropout)
    if config.encoder == "conv_transformer":
        return ConvTransformerEncoder(
            n_landmarks, hands, config.hidden, config.layers, config.embedding_dim, config.dropout, config.heads, config.kernel_size
        )
    raise ValueError(f"unknown encoder {config.encoder}")


def load_run(run_dir: Path, data: PreparedData) -> tuple[TrainConfig, nn.Module]:
    """The config of a training run and its best model (on the CPU)."""
    values = json.loads((run_dir / "config.json").read_text())
    as_tuples = lambda d: {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}  # noqa: E731
    config = TrainConfig(**as_tuples(values) | {"augment": AugmentConfig(**as_tuples(values.get("augment", {})))})
    model = build_model(config, data)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu")["model"])
    return config, model


def phonology_targets(clips: pl.DataFrame, signs: Sequence[str]) -> tuple[np.ndarray, list[int]]:
    """The phonological features of `signs` (columns `phonology.<feature>` of `clips`) as class indices,
    shape (n_signs, n_features), -1 where unknown; and the number of classes of each feature."""
    columns = [column for column in clips.columns if column.startswith("phonology.")]
    table = pl.DataFrame({"sign": list(signs)}).join(
        clips.group_by("sign").agg(pl.col(columns).first()), on="sign", how="left", maintain_order="left"
    )
    targets, n_classes = [], []
    for column in columns:
        index = {value: i for i, value in enumerate(sorted(table[column].drop_nulls().unique()))}
        targets.append([index.get(value, -1) for value in table[column]])
        n_classes.append(len(index))
    return np.array(targets, dtype=np.int64).T.reshape(len(signs), len(columns)), n_classes


@torch.no_grad()
def embed(model: nn.Module, dataset: SignDataset, device: str, batch_size: int = 512) -> np.ndarray:
    """Embeddings of all clips of a (non-augmented) dataset, in order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=4)
    parts = []
    for batch in loader:
        with torch.autocast(device, dtype=torch.bfloat16):
            parts.append(model(batch["frames"].to(device), batch["hands"].to(device), batch["mask"].to(device)).float().cpu())
    return torch.cat(parts).numpy()


def quick_validation(model: nn.Module, val: SignDataset, device: str) -> dict[str, float]:
    """Cheap validation metrics for checks during training: k = 1 and 5, one draw of references, no bootstrap."""
    similarity = cosine_similarity(embed(model, val, device))
    summary, _ = evaluate(similarity, val.data.clips[val.positions], ks=(1, 5), n_draws=1, n_bootstrap=0)
    return {f"{metric}_k{k}": value for k, metric, value in summary.select("k", "metric", "value").iter_rows()}


def plot_curves(metrics: pl.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(metrics["epoch"], metrics["loss"], label="loss")
    ax1.set(xlabel="epoch", ylabel="training loss", title="ArcFace loss")
    ax1b = ax1.twinx()
    ax1b.plot(metrics["epoch"], metrics["train_accuracy"], color="tab:orange", label="train accuracy")
    ax1b.set(ylabel="training accuracy")
    if "val_eer_k1" in metrics.columns:  # after the first validation check
        val = metrics.drop_nulls("val_eer_k1")
        for column, label in [("top1_k1", "top-1, k = 1"), ("top1_k5", "top-1, k = 5"), ("eer_k1", "EER, k = 1"), ("eer_k5", "EER, k = 5")]:
            ax2.plot(val["epoch"], val[f"val_{column}"], marker="o", label=label)
        ax2.legend()
    ax2.set(xlabel="epoch", title="Validation (quick: one draw of references)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(config: TrainConfig, data: PreparedData, run_dir: Path, device: str = "cuda") -> nn.Module:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))

    train_signs = np.random.default_rng(config.seed).permutation(sorted(data.clips.filter(pl.col("split") == "train")["sign"].unique()))
    train_signs = train_signs[: round(config.train_sign_fraction * len(train_signs))]
    train_set = SignDataset(data, "train", config.augment, exclude_signers=config.holdout_signers, only_signs=train_signs)
    val_set = SignDataset(data, "val")
    loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=config.num_workers, persistent_workers=True,
    )  # fmt: skip
    model = build_model(config, data).to(device)
    head = ArcFace(config.embedding_dim, len(train_set.signs), config.arcface_scale, config.arcface_margin).to(device)
    targets, n_classes = phonology_targets(data.clips, train_set.signs) if config.phonology_weight else (np.zeros((0, 0)), [])
    if config.phonology_weight and not n_classes:
        raise ValueError("phonology_weight is set, but the prepared clips have no phonology columns")
    targets = torch.from_numpy(targets).to(device)
    if config.phonology_input not in ("embedding", "pooled"):
        raise ValueError(f"unknown phonology_input {config.phonology_input}")
    head_input = model.head.in_features if config.phonology_input == "pooled" else config.embedding_dim
    phonology_heads = nn.ModuleList(nn.Linear(head_input, n) for n in n_classes).to(device)
    parameters = [*model.parameters(), *head.parameters(), *phonology_heads.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps, warmup_steps = config.epochs * len(loader), config.warmup_epochs * len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, (step + 1) / warmup_steps) * 0.5 * (1 + np.cos(np.pi * min(1, step / total_steps)))
    )

    rows, best_eer = [], np.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        start, losses, phonology_losses, correct, seen = time.time(), [], [], 0, 0
        for batch in loader:
            labels = batch["labels"].to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                embeddings, pooled = model(batch["frames"].to(device), batch["hands"].to(device), batch["mask"].to(device), return_pooled=True)
                logits = head(embeddings, labels)
            loss = arcface_loss = F.cross_entropy(logits, labels)
            if phonology_heads:
                features, inputs = targets[labels], (pooled if config.phonology_input == "pooled" else embeddings).float()
                total = sum(
                    F.cross_entropy(h(inputs), features[:, j], ignore_index=-1, reduction="sum") for j, h in enumerate(phonology_heads)
                )
                phonology_loss = total / (features >= 0).sum().clamp(min=1)  # mean over the known features
                loss = loss + config.phonology_weight * phonology_loss
                phonology_losses.append(phonology_loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(arcface_loss.item())
            correct += (head(embeddings.detach()).argmax(dim=1) == labels).sum().item()
            seen += len(labels)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "train_accuracy": correct / seen, "lr": schedule.get_last_lr()[0]}
        if phonology_losses:
            row["phonology_loss"] = float(np.mean(phonology_losses))
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            val_metrics = quick_validation(model, val_set, device)
            row |= {f"val_{name}": value for name, value in val_metrics.items()}
            if val_metrics["eer_k1"] < best_eer:
                best_eer = val_metrics["eer_k1"]
                torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        row["seconds"] = time.time() - start
        rows.append(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        metrics = pl.DataFrame(rows)
        metrics.write_csv(run_dir / "metrics.csv")
        plot_curves(metrics, run_dir / "curves.png")

    model.load_state_dict(torch.load(run_dir / "best.pt")["model"])
    summary, per_sign = evaluate(cosine_similarity(embed(model, val_set, device)), val_set.data.clips[val_set.positions])
    summary.write_parquet(run_dir / "val_summary.parquet")
    per_sign.write_parquet(run_dir / "val_per_sign.parquet")
    print(summary.pivot(on="metric", index="k", values="value"))
    return model
