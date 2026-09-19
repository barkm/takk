"""Evaluate the self-recorded clips in the setting the system is meant for.

The references are the clips of a glossary, a prepared evaluation set used as the dictionary the
user's attempt is checked against (by default the held-out test signs of ASL Citizen); the queries
are the recordings of its signs. Both sets are embedded with one model and scored against each
other, so a recording's score for a sign is its mean similarity to k clips of that sign by k
different signers of the glossary (`evaluation.evaluate` with a query mask: references come from all
clips, but only the recordings are scored as queries, and references never come from the query's
own signer, which here excludes every recording). A glossary without signer ids, such as Svenskt
teckenspråkslexikon, counts as one signer, so k > 1 still draws a single reference per sign.

Every result comes in two settings:
- copied: the whole glossary, including the clips the recording was shown to copy. This is the
  product: a user copies the dictionary clip and is checked against it.
- uncopied: the glossary without the clips any recording of the sign was shown (`references` in the
  recordings' clips.csv). This asks whether the model recognizes the sign rather than the one
  performance that was copied. A sign with no other clip, like most lexicon entries, has no trial
  here. Recordings made before `references` existed leave nothing out, so for them both settings match.

Signs the glossary knows to be one sign form (its twins.csv) are not scored against each other, as in
`evaluate_test.py`. Beside the pooled metrics it prints each recording with the rank of its own sign
among the whole glossary (twins included) and the signs it scored highest, which is what few clips can
actually say something about.

Run from the repo root: uv run scripts/evaluate_recordings.py --run gru_auslan_phonpool1
    uv run scripts/evaluate_recordings.py --name recordings_sts --glossary data/prepared/sts_lexikon-<id>
Writes outputs/runs/<run>/<name>_{summary,per_sign,per_clip}.parquet, with a `setting` column.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.collection import RAW_DIR, read_clips
from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, draw_scores, evaluate, signer_codes
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
METRICS = ["auc", "eer", "top1", "top5"]


def top_signs(similarity: np.ndarray, clips: pl.DataFrame, queries: np.ndarray, k: int, n: int, seed: int) -> pl.DataFrame:
    """For every query clip, the rank of its own sign and the `n` signs it scored highest."""
    sign_names, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    signers = signer_codes(clips)
    query_rows = np.flatnonzero(queries)
    scores = draw_scores(similarity, signs, signers, k, np.random.default_rng(seed), query_rows)
    own = scores[np.arange(len(query_rows)), signs[query_rows]]
    rows = []
    for row, i in enumerate(query_rows):
        order = np.argsort(-np.nan_to_num(scores[row], nan=-np.inf))
        rows.append(
            {
                "clip_id": clips["clip_id"][int(i)],
                "sign": sign_names[signs[i]],
                "rank": int((scores[row] > own[row]).sum() + 1),
                "score": float(own[row]),
                "top_signs": ", ".join(f"{sign_names[j]} {scores[row][j]:.2f}" for j in order[:n]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="gru_auslan_phonpool1", help="training run whose model embeds both sets")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"recordings-{PrepConfig().id()}")
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--recordings_dir", type=Path, default=RAW_DIR, help="the raw recordings, for the clips each was shown")
    parser.add_argument("--name", default="recordings", help="name of the evaluation, used in the output file names")
    parser.add_argument("--k", type=int, default=5, help="reference clips per sign in the per-clip listing")
    parser.add_argument("--device", default="cpu", help="the GPU is shared; embedding a few thousand clips is cheap on the CPU")
    args = parser.parse_args()

    recordings, dictionary = PreparedData(args.prepared), PreparedData(args.glossary)
    references = SignDataset(dictionary, "test")
    reference_clips = dictionary.clips[references.positions]
    queries = SignDataset(recordings, "test", only_signs=set(reference_clips["sign"]))
    query_clips = recordings.clips[queries.positions]
    if not len(query_clips):
        raise SystemExit(f"no recordings in {args.prepared} of the signs of {args.glossary}")
    shown = read_clips(args.recordings_dir / "clips.csv").filter(pl.col("clip_id").is_in(query_clips["clip_id"].to_list()))["references"]
    shown = {clip for joined in shown.drop_nulls() for clip in joined.split(";") if clip}
    print(f"queries: {len(query_clips)} recordings, {query_clips['sign'].n_unique()} signs, {query_clips['signer'].n_unique()} signers")
    print(f"glossary: {len(reference_clips)} clips, {reference_clips['sign'].n_unique()} signs, {reference_clips['signer'].n_unique()} signers; {len(shown)} of the clips were shown")

    _, model = load_run(RUNS_DIR / args.run, dictionary)
    model = model.to(args.device)
    embeddings = np.concatenate([embed(model, queries, args.device), embed(model, references, args.device)])
    columns = ["clip_id", "sign", "signer"]
    clips = pl.concat([query_clips.select(columns), reference_clips.select(columns)])
    is_query = np.arange(len(clips)) < len(query_clips)
    similarity = cosine_similarity(embeddings)

    tables: dict[str, list[pl.DataFrame]] = {"summary": [], "per_sign": [], "per_clip": []}
    uncopied = ~(~is_query & clips["clip_id"].is_in(list(shown)).to_numpy())
    for setting, keep in [("copied", np.ones(len(clips), dtype=bool)), ("uncopied", uncopied)]:
        kept = clips.filter(keep)
        if not kept.filter(~is_query[keep])["sign"].is_in(query_clips["sign"].to_list()).any():
            print(f"\n{setting}: no recording's sign has a glossary clip left")
            continue
        kept_similarity = similarity if keep.all() else similarity[np.ix_(keep, keep)]
        summary, per_sign = evaluate(kept_similarity, kept, queries=is_query[keep], twins=dictionary.twins)
        per_clip = top_signs(kept_similarity, kept, is_query[keep], args.k, 5, seed=0)
        for name, table in [("summary", summary), ("per_sign", per_sign), ("per_clip", per_clip)]:
            tables[name].append(table.with_columns(setting=pl.lit(setting)))

        print(f"\n{args.run}, {setting}, glossary {args.glossary.name}:")
        for k, group in summary.group_by("k", maintain_order=True):
            values = {m: f"{v:.3f} [{low:.3f}, {high:.3f}]" for m, v, low, high in group.select("metric", "value", "ci_low", "ci_high").iter_rows()}
            print(f"  k={k[0]}: " + "  ".join(f"{m} {values[m]}" for m in METRICS))

    out = RUNS_DIR / args.run
    for name, parts in tables.items():
        pl.concat(parts).write_parquet(out / f"{args.name}_{name}.parquet")
    per_clip = pl.concat(tables["per_clip"])
    print(f"\nper recording (k={args.k}), rank of its own sign among {clips['sign'].n_unique()} signs:")
    with pl.Config(fmt_str_lengths=120, tbl_rows=-1, tbl_width_chars=220):
        print(per_clip.filter(pl.col("setting") == "copied").drop("setting").join(
            per_clip.filter(pl.col("setting") == "uncopied").select("clip_id", rank_uncopied="rank"), on="clip_id", how="left"
        ).sort("rank"))


if __name__ == "__main__":
    main()
