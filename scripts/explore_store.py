"""Exploratory statistics and figures for a landmark store of any dataset.

Run from the repo root: uv run scripts/explore_store.py data/processed/asl_citizen
Figures go to outputs/eda/<store name>/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless machine: figures are saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from tqdm import tqdm

from isolated_sign_validation.landmarks import LANDMARK_GROUPS, LANDMARK_SLICES, LandmarkStore

CHUNK_FRAMES = 200_000
COORD_SAMPLE_STEP = 20  # every n-th frame is used for the coordinate ranges


def read_frames(store: LandmarkStore) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Per-frame presence of the left and right hand, and x, y of a sample of frames per landmark group."""
    n = len(store.landmarks)
    left, right = np.empty(n, dtype=bool), np.empty(n, dtype=bool)
    groups = ["left_hand", "right_hand", "upper_body"]
    coords = {group: [] for group in groups}
    for start in tqdm(range(0, n, CHUNK_FRAMES), desc="reading landmarks", unit="chunk"):
        block = np.asarray(store.landmarks[start : start + CHUNK_FRAMES])
        # A hand is either detected with all its landmarks or fully NaN, so its wrist tells if it is present.
        left[start : start + len(block)] = ~np.isnan(block[:, LANDMARK_SLICES["left_hand"].start, 0])
        right[start : start + len(block)] = ~np.isnan(block[:, LANDMARK_SLICES["right_hand"].start, 0])
        for group in groups:
            coords[group].append(block[::COORD_SAMPLE_STEP][:, LANDMARK_GROUPS[group], :2].reshape(-1, 2))
    return left, right, {group: np.concatenate(c) for group, c in coords.items()}


def clip_stats(store: LandmarkStore, left: np.ndarray, right: np.ndarray) -> pl.DataFrame:
    starts, n_frames = store.clips["offset"].to_numpy(), store.clips["n_frames"].to_numpy()
    if (n_frames == 0).any():
        raise ValueError("store has empty clips")
    count = lambda frames: np.add.reduceat(frames.astype(np.int64), starts)  # noqa: E731
    any_hand = left | right
    active = np.empty(len(starts), dtype=np.int64)  # frames from the first to the last frame with a hand
    for i, (start, n) in enumerate(zip(starts, n_frames)):
        idx = np.flatnonzero(any_hand[start : start + n])
        active[i] = idx[-1] - idx[0] + 1 if len(idx) else 0
    return store.clips.with_columns(
        duration=pl.col("n_frames") / pl.col("fps"),
        left_frames=count(left),
        right_frames=count(right),
        both_frames=count(left & right),
        hand_frames=count(any_hand),
        active_frames=active,
    ).with_columns(
        active_duration=pl.col("active_frames") / pl.col("fps"),
        dominant=pl.when(pl.col("left_frames") > pl.col("right_frames")).then(pl.lit("left")).otherwise(pl.lit("right")),
    )


def save(fig: Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / name}")


def report(stats: pl.DataFrame, coords: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = stats["n_frames"].sum()
    print(f"{stats.height} clips, {frames} frames, {stats['sign'].n_unique()} signs, {stats['signer'].n_unique()} signers")
    print("fps (rounded):", stats["fps"].round(0).value_counts(sort=True).head(8).rows())

    q = lambda col: {p: round(stats[col].quantile(p), 2) for p in (0.01, 0.5, 0.95, 0.99)} | {"max": stats[col].max()}  # noqa: E731
    print("\nclip duration (s):", q("duration"))
    print("active duration (s), first to last frame with a hand:", q("active_duration"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["duration", "active_duration"], ["Clip duration", "First to last frame with a hand"]):
        ax.hist(stats[col].clip(upper_bound=10), bins=100)
        ax.set(xlabel="seconds (clipped at 10)", ylabel="clips", title=title, yscale="log")
    save(fig, out_dir, "durations.png")

    hand_frames, active_frames = stats["hand_frames"].sum(), stats["active_frames"].sum()
    print(f"\nclips without any hand: {(stats['hand_frames'] == 0).sum()}")
    print(
        f"frames without any hand: {1 - hand_frames / frames:.1%} = {1 - active_frames / frames:.1%} before the first / "
        f"after the last hand + {(active_frames - hand_frames) / frames:.1%} gaps during signing"
    )
    both_share = stats["both_frames"] / stats["hand_frames"]
    print(f"clips with both hands in at least half of their hand frames: {(both_share >= 0.5).mean():.1%}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(1 - stats["hand_frames"] / stats["n_frames"], bins=50)
    axes[0].set(xlabel="fraction of frames without any hand", ylabel="clips", title="Frames without hands per clip")
    axes[1].hist(both_share.drop_nans(), bins=50)
    axes[1].set(xlabel="frames with both hands / frames with a hand", ylabel="clips", title="Two-handedness per clip")
    save(fig, out_dir, "hands.png")

    per_signer = (
        stats.group_by("signer")
        .agg(clips=pl.len(), left_dominant=(pl.col("dominant") == "left").mean())
        .sort("clips", descending=True)
    )
    print(f"\nclips with left_hand dominant: {(stats['dominant'] == 'left').mean():.1%}")
    print("signers by share of clips with left_hand dominant:")
    print(per_signer["left_dominant"].cut([0.1, 0.5, 0.9]).value_counts().sort("left_dominant"))
    labels = per_signer["signer"].str.split(":").list.last()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(labels, per_signer["clips"])
    ax1.set(ylabel="clips", title="Clips per signer")
    ax2.bar(labels, per_signer["left_dominant"])
    ax2.set(ylabel="share", title="Share of clips where left_hand is detected more often than right_hand")
    ax2.tick_params(axis="x", rotation=90)
    save(fig, out_dir, "signers.png")

    print("\ncoordinate percentiles (0.1%, 1%, 99%, 99.9%) of a frame sample:")
    for group, xy in coords.items():
        xy = xy[~np.isnan(xy[:, 0])]
        for axis, values in zip("xy", xy.T):
            print(f"  {group:10s} {axis}: {np.round(np.percentile(values, [0.1, 1, 99, 99.9]), 2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    args = parser.parse_args()

    store = LandmarkStore(args.store)
    left, right, coords = read_frames(store)
    report(clip_stats(store, left, right), coords, Path("outputs/eda") / args.store.name)


if __name__ == "__main__":
    main()
