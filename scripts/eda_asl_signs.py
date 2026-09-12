"""Exploratory data analysis of the Kaggle ASL Signs dataset.

Computes per-clip statistics (cached in data/interim/) and writes figures to outputs/eda/.
Run from the repo root: uv run scripts/eda_asl_signs.py
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")  # parallelism comes from worker processes

import matplotlib

matplotlib.use("Agg")  # headless machine: figures are saved to files, never shown
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure
from tqdm import tqdm

DATA_DIR = Path("data/raw/asl-signs")
STATS_PATH = Path("data/interim/asl_signs_clip_stats.parquet")
OUT_DIR = Path("outputs/eda")


def hand_present(hand: str) -> pl.Expr:
    return ((pl.col("type") == hand) & pl.col("x").is_not_null()).any()


def clip_stats(path: str) -> dict:
    df = pl.read_parquet(DATA_DIR / path, columns=["frame", "type", "x", "y"])
    frames = df.group_by("frame").agg(left=hand_present("left_hand"), right=hand_present("right_hand"))
    return {
        "path": path,
        "n_frames": frames.height,
        "frame_span": frames.select(pl.col("frame").max() - pl.col("frame").min() + 1).item(),
        "left_frac": frames["left"].mean(),
        "right_frac": frames["right"].mean(),
        "no_hand_frac": (~(frames["left"] | frames["right"])).mean(),
        "x_min": df["x"].min(),
        "x_max": df["x"].max(),
        "y_min": df["y"].min(),
        "y_max": df["y"].max(),
    }


def compute_stats(train: pl.DataFrame) -> pl.DataFrame:
    if STATS_PATH.exists():
        return pl.read_parquet(STATS_PATH)
    ctx = multiprocessing.get_context("spawn")  # polars is not fork-safe
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as pool:
        results = pool.map(clip_stats, train["path"], chunksize=64)
        rows = list(tqdm(results, total=train.height, desc="clip stats", unit="clip"))
    stats = train.join(pl.DataFrame(rows), on="path")
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.write_parquet(STATS_PATH)
    return stats


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR / name}")


def plot_all(stats: pl.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_sign = stats["sign"].value_counts().sort("count", descending=True)
    print(f"\n{per_sign.height} signs, clips per sign: {per_sign['count'].describe()}")
    print("fewest clips:", per_sign.tail(3).rows(), " most clips:", per_sign.head(3).rows())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(per_sign.height), per_sign["count"])
    ax.set(xlabel="sign (sorted by count)", ylabel="clips", title=f"Clips per sign ({per_sign.height} signs)")
    save(fig, "clips_per_sign.png")

    per_part = (
        stats.group_by("participant_id")
        .agg(clips=pl.len(), left=pl.col("left_frac").mean(), right=pl.col("right_frac").mean())
        .sort("clips", descending=True)
    )
    print(f"\n{per_part.height} participants:\n{per_part}")
    labels = per_part["participant_id"].cast(pl.String)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.bar(labels, per_part["clips"])
    ax1.set(ylabel="clips", title="Clips per participant")
    x = range(per_part.height)
    ax2.bar([i - 0.2 for i in x], per_part["left"], width=0.4, label="left_hand")
    ax2.bar([i + 0.2 for i in x], per_part["right"], width=0.4, label="right_hand")
    ax2.set(ylabel="mean fraction of frames", title="Hand presence per participant")
    ax2.set_xticks(list(x), labels, rotation=90)
    ax2.legend()
    save(fig, "participants.png")

    print(f"\nframes per clip: {stats['n_frames'].describe()}")
    print("percentiles:", {q: stats["n_frames"].quantile(q) for q in (0.05, 0.5, 0.95, 0.99)})
    print("clips with gaps in frame numbers:", (stats["frame_span"] != stats["n_frames"]).sum())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["n_frames"], bins=100)
    ax.set(xlabel="frames", ylabel="clips", title="Sequence length", yscale="log")
    save(fig, "sequence_length.png")

    print("\nclips with no hand in any frame:", (stats["no_hand_frac"] == 1).sum())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["no_hand_frac"], bins=50)
    ax.set(xlabel="fraction of frames with no hand detected", ylabel="clips", title="Missing hands", yscale="log")
    save(fig, "missing_hands.png")

    coords = ["x_min", "x_max", "y_min", "y_max"]
    print(f"\ncoordinate ranges:\n{stats.select(coords).describe()}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, col in zip(axes, coords):
        ax.hist(stats[col].drop_nulls(), bins=100)
        ax.set(title=col, yscale="log")
    save(fig, "coordinate_ranges.png")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, row in zip(axes, stats.sample(4, seed=0).iter_rows(named=True)):
        df = pl.read_parquet(DATA_DIR / row["path"])
        frame = df.filter(pl.col("frame") == df["frame"].sort()[df["frame"].len() // 2])
        for type_ in ["face", "pose", "left_hand", "right_hand"]:
            group = frame.filter(pl.col("type") == type_).drop_nulls("x")
            ax.scatter(group["x"], group["y"], s=2 if type_ == "face" else 8, label=type_)
        ax.invert_yaxis()  # image coordinates: y grows downward
        ax.set(title=f"{row['sign']} (participant {row['participant_id']})", aspect="equal")
    axes[0].legend(markerscale=2)
    save(fig, "sample_frames.png")


if __name__ == "__main__":
    train = pl.read_csv(DATA_DIR / "train.csv")
    plot_all(compute_stats(train))
