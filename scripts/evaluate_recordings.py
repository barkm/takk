"""Evaluate the self-recorded clips in the setting the system is meant for.

The references are ASL Citizen clips of the held-out test signs, as a dictionary the user's attempt
is checked against; the queries are the recordings. Both sets are embedded with one model and scored
against each other, so a recording's score for a sign is its mean similarity to k clips of that sign
by k different ASL Citizen signers (`evaluation.evaluate` with a query mask: references come from
all clips, but only the recordings are scored as queries, and references never come from the query's
own signer, which here excludes every recording).

Beside the pooled metrics it prints each recording with the rank of its own sign among the whole
glossary and the signs it scored highest, which is what few clips can actually say something about.

Run from the repo root: uv run scripts/evaluate_recordings.py --run gru_auslan_phonpool1
Writes outputs/runs/<run>/recordings_{summary,per_sign,per_clip}.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, draw_scores, evaluate
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
METRICS = ["auc", "eer", "top1", "top5"]


def top_signs(similarity: np.ndarray, clips: pl.DataFrame, queries: np.ndarray, k: int, n: int, seed: int) -> pl.DataFrame:
    """For every query clip, the rank of its own sign and the `n` signs it scored highest."""
    sign_names, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    _, signers = np.unique(clips["signer"].to_numpy(), return_inverse=True)
    scores = draw_scores(similarity, signs, signers, k, np.random.default_rng(seed))
    own = scores[np.arange(len(signs)), signs]
    rows = []
    for i in np.flatnonzero(queries):
        order = np.argsort(-np.nan_to_num(scores[i], nan=-np.inf))
        rows.append(
            {
                "clip_id": clips["clip_id"][int(i)],
                "sign": sign_names[signs[i]],
                "rank": int((scores[i] > own[i]).sum() + 1),
                "score": float(own[i]),
                "top_signs": ", ".join(f"{sign_names[j]} {scores[i][j]:.2f}" for j in order[:n]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="gru_auslan_phonpool1", help="training run whose model embeds both sets")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"recordings-{PrepConfig().id()}")
    parser.add_argument("--references", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--k", type=int, default=5, help="reference clips per sign in the per-clip listing")
    parser.add_argument("--device", default="cpu", help="the GPU is shared; embedding a few thousand clips is cheap on the CPU")
    args = parser.parse_args()

    recordings, dictionary = PreparedData(args.prepared), PreparedData(args.references)
    queries, references = SignDataset(recordings, "test"), SignDataset(dictionary, "test")
    query_clips, reference_clips = recordings.clips[queries.positions], dictionary.clips[references.positions]
    print(f"queries: {len(query_clips)} recordings, {query_clips['sign'].n_unique()} signs, {query_clips['signer'].n_unique()} signers")
    print(f"glossary: {len(reference_clips)} ASL Citizen clips, {reference_clips['sign'].n_unique()} signs, {reference_clips['signer'].n_unique()} signers")

    _, model = load_run(RUNS_DIR / args.run, dictionary)
    model = model.to(args.device)
    embeddings = np.concatenate([embed(model, queries, args.device), embed(model, references, args.device)])
    columns = ["clip_id", "sign", "signer"]
    clips = pl.concat([query_clips.select(columns), reference_clips.select(columns)])
    is_query = np.arange(len(clips)) < len(query_clips)

    similarity = cosine_similarity(embeddings)
    summary, per_sign = evaluate(similarity, clips, queries=is_query)
    per_clip = top_signs(similarity, clips, is_query, args.k, 5, seed=0)
    out = RUNS_DIR / args.run
    for name, table in [("summary", summary), ("per_sign", per_sign), ("per_clip", per_clip)]:
        table.write_parquet(out / f"recordings_{name}.parquet")

    print(f"\n{args.run}, references by ASL Citizen signers:")
    for k, group in summary.group_by("k", maintain_order=True):
        values = {m: f"{v:.3f} [{low:.3f}, {high:.3f}]" for m, v, low, high in group.select("metric", "value", "ci_low", "ci_high").iter_rows()}
        print(f"  k={k[0]}: " + "  ".join(f"{m} {values[m]}" for m in METRICS))
    print(f"\nper recording (k={args.k}), rank of its own sign among {clips['sign'].n_unique()} signs:")
    with pl.Config(fmt_str_lengths=120, tbl_rows=-1, tbl_width_chars=200):
        print(per_clip.sort("rank"))


if __name__ == "__main__":
    main()
