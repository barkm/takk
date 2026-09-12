"""Render landmark clips side by side as an animated GIF, for inspecting the data.

By default picks clips from different signers. Examples, from the repo root:
    uv run scripts/view_clips.py --sign hello            # 4 clips of "hello" by different signers
    uv run scripts/view_clips.py --signer 26734 -n 2     # 2 random clips by one signer
    uv run scripts/view_clips.py --clip-id 1000035562
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

from isolated_sign_validation.landmarks import LANDMARK_GROUPS, LANDMARK_SLICES, LandmarkStore

# MediaPipe HandLandmarksConnections.HAND_CONNECTIONS
HAND_EDGES = [(0, 1), (0, 17), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (5, 9), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (13, 17), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]  # fmt: skip
# MediaPipe PoseLandmarksConnections.POSE_LANDMARKS between shoulders, arms and pose hand points
UPPER_BODY_EDGES = [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20)]  # fmt: skip
# MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_LIPS
LIPS_EDGES = [(0, 267), (13, 312), (14, 317), (17, 314), (37, 0), (39, 37), (40, 39), (61, 146), (61, 185), (78, 95), (78, 191), (80, 81), (81, 82), (82, 13), (84, 17), (87, 14), (88, 178), (91, 181), (95, 88), (146, 91), (178, 87), (181, 84), (185, 40), (191, 80), (267, 269), (269, 270), (270, 409), (310, 415), (311, 310), (312, 311), (314, 405), (317, 402), (318, 324), (321, 375), (324, 308), (375, 291), (402, 318), (405, 321), (409, 291), (415, 308)]  # fmt: skip

# Skeleton lines to draw: (edges as indices into the landmark axis, color)
SKELETON = [
    (np.array(HAND_EDGES) + LANDMARK_SLICES["left_hand"].start, "tab:blue"),
    (np.array(HAND_EDGES) + LANDMARK_SLICES["right_hand"].start, "tab:red"),
    (np.array(UPPER_BODY_EDGES) + LANDMARK_SLICES["pose"].start, "tab:gray"),
    (np.array(LIPS_EDGES) + LANDMARK_SLICES["face"].start, "tab:pink"),
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


def render(store: LandmarkStore, clips: pl.DataFrame, out: Path, fps: int) -> None:
    fig, axes = plt.subplots(1, clips.height, figsize=(4 * clips.height, 5), squeeze=False)
    face = np.arange(LANDMARK_SLICES["face"].start, LANDMARK_SLICES["face"].stop)
    drawn = np.concatenate([face, *LANDMARK_GROUPS.values()])  # determines the axis limits
    panels = []
    for ax, row in zip(axes[0], clips.iter_rows(named=True)):
        xy = np.asarray(store[row["row"]][..., :2])
        lo, hi = np.nanmin(xy[:, drawn], axis=(0, 1)), np.nanmax(xy[:, drawn], axis=(0, 1))
        pad = 0.05 * (hi - lo)
        ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(hi[1] + pad[1], lo[1] - pad[1]), aspect="equal")
        ax.set_title(f"{row['sign']}\n{row['signer']}\nclip {row['clip_id']}", fontsize=9)
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
    n_frames = int(clips["n_frames"].max())
    FuncAnimation(fig, update, frames=n_frames, blit=True).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=Path("data/processed/kaggle_asl_signs"))
    parser.add_argument("--sign")
    parser.add_argument("--signer", help="full signer id, or the id without the dataset prefix")
    parser.add_argument("--clip-id")
    parser.add_argument("-n", type=int, default=4, help="number of clips side by side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15, help="playback speed")
    parser.add_argument("--out", type=Path, help="default: outputs/clips/<filters>.gif")
    args = parser.parse_args()

    store = LandmarkStore(args.store)
    clips = select_clips(store.clips, args)
    print(clips.select("clip_id", "sign", "signer", "n_frames"))
    name = "_".join(str(v) for v in (args.sign, args.signer, args.clip_id) if v) or "random"
    out = args.out or Path("outputs/clips") / f"{name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(store, clips, out, args.fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
