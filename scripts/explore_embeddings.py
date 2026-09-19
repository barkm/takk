"""Web app for exploring a trained model's embedding space, in the spirit of https://asl-lex.org/visualization/.

Each point is a sign: the normalized mean of its clips' embeddings. The signs are laid out in 2D by
t-SNE on cosine distance and grouped into clusters (Ward linkage on the normalized sign embeddings),
so signs the model finds similar sit together. Points can be colored by cluster, dataset, split or any
ASL-LEX phonological feature, to see whether the embedding groups by handshape, location, movement...
Clicking a sign shows clips of it, its nearest signs by cosine similarity and the rest of its cluster.

Run from the repo root, then open http://localhost:8001 (forward the port when the machine is remote):
    uv run scripts/explore_embeddings.py --run gru_auslan_phonpool1
    uv run scripts/explore_embeddings.py --prepared data/prepared/asl_citizen-d314433a data/prepared/slovo-d314433a
Embedding every clip runs the model on the GPU for a few seconds; the layout takes about half a minute.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from isolated_sign_validation.collection import WEB_DIR, video_paths
from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
NEIGHBORS = 12  # nearest signs listed per sign
SHOWN_CLIPS = 3  # clips shown per sign, by different signers where possible


def sign_table(clips: pl.DataFrame, embeddings: np.ndarray, n_clusters: int, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """One row per sign (its dataset, split, clip count, phonology and shown clips) with its 2D
    position and cluster, and the signs' normalized mean embeddings in the same order."""
    phonology = [column for column in clips.columns if column.startswith("phonology.")]
    signs = (
        clips.with_row_index("row")
        .sort("signer", "clip_id", nulls_last=True)
        .group_by("sign")
        .agg(
            pl.col("dataset", "split", *phonology).first(),
            pl.len().alias("n_clips"),
            pl.col("row"),
            pl.col("clip_id").unique(maintain_order=True).head(SHOWN_CLIPS).alias("shown"),
            pl.col("clip_id").filter(pl.col("signer").is_first_distinct()).head(SHOWN_CLIPS).alias("by_signer"),
        )
        .sort("sign")
    )
    # prefer clips by different signers; datasets without signer ids fall back to any clips
    signs = signs.with_columns(shown=pl.when(pl.col("by_signer").list.len() > 1).then("by_signer").otherwise("shown")).drop("by_signer")
    means = np.stack([embeddings[rows].mean(axis=0) for rows in signs["row"]])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    xy = TSNE(metric="cosine", init="pca", random_state=seed).fit_transform(means)
    cluster = AgglomerativeClustering(n_clusters=min(n_clusters, len(means))).fit_predict(means)
    return signs.drop("row").with_columns(x=xy[:, 0], y=xy[:, 1], cluster=cluster), means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="gru_auslan_phonpool1")
    parser.add_argument("--prepared", type=Path, nargs="+", default=[Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="splits whose clips are shown")
    parser.add_argument("--clusters", type=int, default=150, help="number of clusters")
    parser.add_argument("--seed", type=int, default=0, help="seeds the layout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    data = PreparedData(*args.prepared)
    shown = pl.when(pl.col("split").is_in(args.splits)).then(pl.lit("explored"))  # the split SignDataset selects
    data.clips = data.clips.with_columns(original_split=pl.col("split"), split=shown)
    clips_set = SignDataset(data, "explored")
    clips = data.clips[clips_set.positions].with_columns(split=pl.col("original_split"))
    print(f"{clips.height} clips of {clips['sign'].n_unique()} signs from {', '.join(p.name for p in args.prepared)}")
    _, model = load_run(RUNS_DIR / args.run, data)
    embeddings = embed(model.to(args.device), clips_set, args.device)
    signs, means = sign_table(clips, embeddings, args.clusters, args.seed)

    similarity = means @ means.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1)[:, :NEIGHBORS]
    points = signs.to_dicts()
    for i, point in enumerate(points):
        point["neighbors"] = [[int(j), round(float(similarity[i, j]), 3)] for j in nearest[i]]  # indices into points
    videos = video_paths(clips.filter(pl.col("clip_id").is_in([clip for shown in signs["shown"] for clip in shown])))
    print(f"{len(points)} signs laid out in {signs['cluster'].n_unique()} clusters; serving http://{args.host}:{args.port}")

    app = FastAPI()
    payload = {"run": args.run, "points": points, "features": ["cluster", "dataset", "split"] + [c for c in clips.columns if c.startswith("phonology.")]}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "embeddings.html")

    @app.get("/api/data")
    def get_data() -> dict:
        return payload

    @app.get("/api/video/{clip_id}")
    def video(clip_id: str) -> FileResponse:
        if clip_id not in videos:
            raise HTTPException(404, "unknown clip")
        return FileResponse(videos[clip_id])

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
