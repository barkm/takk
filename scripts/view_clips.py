"""Render landmark clips side by side as an animated GIF, for inspecting the data.

By default picks clips from different signers. Examples, from the repo root:
    uv run scripts/view_clips.py --sign HELLO            # 4 clips of HELLO by different signers
    uv run scripts/view_clips.py --signer P12 -n 2       # 2 random clips by one signer
    uv run scripts/view_clips.py --clip-id 15890366051589533-APPLE
    uv run scripts/view_clips.py --prepared data/prepared/asl_citizen-b7bd1b06 --sign HELLO
With --prepared, prepared clips (bottom row) are shown below the store clips they come from (top row);
--augment adds a row with a random training augmentation of each prepared clip.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless machine: figures are saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from isolated_sign_validation.dataset import AugmentConfig, augment
from isolated_sign_validation.landmarks import (
    LANDMARK_GROUPS,
    LANDMARK_SLICES,
    N_LANDMARKS,
    SKELETON_EDGES,
    LandmarkStore,
)
from isolated_sign_validation.preparation import PrepConfig, PreparedData

# Skeleton lines to draw: (edges as indices into the landmark axis, color)
SKELETON = [
    (SKELETON_EDGES["left_hand"], "tab:blue"),
    (SKELETON_EDGES["right_hand"], "tab:red"),
    (SKELETON_EDGES["upper_body"], "tab:gray"),
    (SKELETON_EDGES["lips"], "tab:pink"),
]


def select_clips(clips: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    clips = clips.with_row_index("row")
    if args.clip_id:
        clips = clips.filter(pl.col("clip_id") == args.clip_id)
    if args.sign:
        clips = clips.filter(pl.col("sign") == args.sign)
    if args.signer:
        clips = clips.filter((pl.col("signer") == args.signer) | pl.col("signer").str.ends_with(f":{args.signer}"))
    if clips.is_empty():
        raise SystemExit("no clips match the given filters")
    # Shuffle, then take one clip per signer before taking a second clip from any signer.
    return (
        clips.sample(fraction=1, shuffle=True, seed=args.seed)
        .with_columns(repeat=pl.int_range(pl.len()).over("signer"))
        .sort("repeat", maintain_order=True)
        .head(args.n)
    )


def to_landmark_layout(frames: np.ndarray, config: PrepConfig) -> np.ndarray:
    """Prepared frames as x, y in the store's landmark layout; landmarks that were not prepared are NaN."""
    xy = np.full((len(frames), N_LANDMARKS, 2), np.nan, dtype=np.float32)
    xy[:, config.landmarks] = frames[..., :2]
    return xy


def title(row: dict) -> str:
    return f"{row['sign']}\n{row['signer']}\nclip {row['clip_id']}"


def render(rows: list[list[tuple[np.ndarray, str]]], out: Path, fps: int) -> None:
    """Animate rows of panels, each (x, y in the store's landmark layout, title)."""
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4 * n_cols, 5 * len(rows)), squeeze=False)
    face = np.arange(LANDMARK_SLICES["face"].start, LANDMARK_SLICES["face"].stop)
    drawn = np.concatenate([face, *LANDMARK_GROUPS.values()])  # determines the axis limits
    panels = []
    for ax, (xy, panel_title) in [(ax, panel) for ax_row, row in zip(axes, rows) for ax, panel in zip(ax_row, row)]:
        lo, hi = np.nanmin(xy[:, drawn], axis=(0, 1)), np.nanmax(xy[:, drawn], axis=(0, 1))
        pad = 0.05 * (hi - lo)
        ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(hi[1] + pad[1], lo[1] - pad[1]), aspect="equal")
        ax.set_title(panel_title, fontsize=9)
        ax.tick_params(labelsize=7)
        mesh = ax.plot([], [], "o", ms=1, color="lightgray")[0]
        reference = ax.plot([], [], "o", ms=3, color="black")[0]
        lines = [ax.add_collection(LineCollection([], colors=color, linewidths=1.5)) for _, color in SKELETON]
        label = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8)
        panels.append((xy, mesh, reference, lines, label))

    def update(t: int) -> list:
        artists = []
        for xy, mesh, reference, lines, label in panels:
            i = min(t, len(xy) - 1)  # shorter clips hold their last frame
            mesh.set_data(xy[i, face, 0], xy[i, face, 1])
            reference.set_data(xy[i, LANDMARK_GROUPS["face_reference"], 0], xy[i, LANDMARK_GROUPS["face_reference"], 1])
            for line, (edges, _) in zip(lines, SKELETON):
                line.set_segments(xy[i][edges])
            label.set_text(f"frame {i + 1}/{len(xy)}")
            artists += [mesh, reference, *lines, label]
        return artists

    fig.tight_layout()
    n_frames = max(len(xy) for xy, *_ in panels)
    FuncAnimation(fig, update, frames=n_frames, blit=True).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=Path("data/processed/asl_citizen"))
    parser.add_argument("--prepared", type=Path, help="prepared data made from --store; show its clips too")
    parser.add_argument("--augment", action="store_true", help="with --prepared, also show augmented clips")
    parser.add_argument("--sign")
    parser.add_argument("--signer", help="full signer id, or the id without the dataset prefix")
    parser.add_argument("--clip-id")
    parser.add_argument("-n", type=int, default=4, help="number of clips side by side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15, help="playback speed")
    parser.add_argument("--out", type=Path, help="default: outputs/clips/<filters>.gif")
    args = parser.parse_args()

    store = LandmarkStore(args.store)
    if args.prepared:
        data = PreparedData(args.prepared)
        clips = select_clips(data.clips, args)
        rows = [
            [(np.asarray(store[row["store_row"]][..., :2]), title(row)) for row in clips.iter_rows(named=True)],
            [
                (to_landmark_layout(data[row["row"]], data.config), f"prepared ({row['split']})" + ", mirrored" * (row["dominant"] == "left"))
                for row in clips.iter_rows(named=True)
            ],
        ]
        if args.augment:
            rng = np.random.default_rng(args.seed)
            hands = [data.config.group_slices[hand] for hand in ("left_hand", "right_hand")]
            augmented = [augment(data[row["row"]], rng, AugmentConfig(), hands, data.config.max_frames) for row in clips.iter_rows(named=True)]
            rows.append([(to_landmark_layout(frames, data.config), "augmented") for frames in augmented])
    else:
        clips = select_clips(store.clips, args)
        rows = [[(np.asarray(store[row["row"]][..., :2]), title(row)) for row in clips.iter_rows(named=True)]]
    print(clips.select("clip_id", "sign", "signer", "n_frames"))
    name = "_".join(str(v) for v in (args.sign, args.signer, args.clip_id) if v) or "random"
    out = args.out or Path("outputs/clips") / f"{name}{'_prepared' * bool(args.prepared)}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(rows, out, args.fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
