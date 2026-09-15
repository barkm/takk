"""False accepts at a fixed rate of false rejects: for random wrong signs and for similar ones.

For each run, the global threshold that rejects a given share of correct attempts (k references by
other signers, 5 draws), and at that threshold the share of accepted attempts of a wrong sign: a random
one, the target's most similar sign (by the run's own sign embeddings, the mean of a sign's clips), and
phonological near-minimal pairs (signs whose ASL-LEX features differ in at most --max-differences of
them; the same pairs for every run). 95% bootstrap CIs over the attempted signs, at a fixed threshold.

Run from the repo root: uv run scripts/operating_points.py gru_auslan gru_auslan_phon1
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, draw_scores
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")


def accept_rate(accepted: np.ndarray, attempted_signs: np.ndarray, n_signs: int, rng: np.random.Generator) -> str:
    """Share of accepted attempts, with a 95% bootstrap CI over the attempted signs."""
    hits, counts = np.bincount(attempted_signs, accepted, n_signs), np.bincount(attempted_signs, minlength=n_signs)
    weights = np.stack([np.bincount(rng.integers(0, n_signs, n_signs), minlength=n_signs) for _ in range(1000)])
    low, high = np.percentile(weights @ hits / (weights @ counts), [2.5, 97.5])
    return f"{hits.sum() / counts.sum():.3f} [{low:.3f}, {high:.3f}]"


def near_minimal_pairs(clips: pl.DataFrame, sign_names: np.ndarray, max_differences: int) -> np.ndarray:
    """Boolean matrix over signs: pairs whose phonological features (all known) differ in at most max_differences."""
    columns = [column for column in clips.columns if column.startswith("phonology.")]
    table = clips.group_by("sign").agg(pl.col(columns).first()).sort("sign")
    assert table["sign"].to_list() == list(sign_names)
    values = table.select(columns).to_numpy()
    known = ~table.select(pl.any_horizontal(pl.col(columns).is_null())).to_series().to_numpy()
    differences = (values[:, None] != values[None]).sum(axis=2)
    pairs = (differences <= max_differences) & known[:, None] & known[None]
    np.fill_diagonal(pairs, False)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared") / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--rejected", type=float, default=0.05, help="share of correct attempts rejected")
    parser.add_argument("--max-differences", type=int, default=2)
    args = parser.parse_args()

    data = PreparedData(args.prepared)
    dataset = SignDataset(data, args.split)
    clips = data.clips[dataset.positions]
    sign_names, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    _, signers = np.unique(clips["signer"].to_numpy(), return_inverse=True)
    minimal = near_minimal_pairs(clips, sign_names, args.max_differences)
    print(f"{args.split}: {len(sign_names)} signs, {minimal.sum() // 2} near-minimal pairs (at most {args.max_differences} of the "
          f"ASL-LEX features differ) among {minimal.any(axis=1).sum()} signs; threshold rejects {args.rejected:.0%} of correct attempts")  # fmt: skip

    rows = []
    for run in args.runs:
        _, model = load_run(RUNS_DIR / run, data)
        embeddings = embed(model.to("cuda"), dataset, "cuda")
        similarity = cosine_similarity(embeddings)
        means = np.stack([embeddings[signs == s].mean(axis=0) for s in range(len(sign_names))])
        means /= np.linalg.norm(means, axis=1, keepdims=True)
        between = means @ means.T
        np.fill_diagonal(between, -np.inf)
        nearest = between.argmax(axis=1)
        for k in (1, 5):
            rng = np.random.default_rng(0)
            attempts = {"correct": [], "random wrong sign": [], "most similar sign": [], "near-minimal pair": []}  # (score, attempted sign)
            for _ in range(5):
                scores = draw_scores(similarity, signs, signers, k, rng)
                valid = ~np.isnan(scores[np.arange(len(signs)), signs])
                wrong = np.ones(scores.shape, dtype=bool)
                wrong[np.arange(len(signs)), signs] = False
                for name, pairs in [
                    ("correct", ~wrong), ("random wrong sign", wrong),
                    ("most similar sign", np.eye(len(sign_names), dtype=bool)[nearest[signs]]), ("near-minimal pair", minimal[signs]),
                ]:  # fmt: skip
                    query, target = np.nonzero(pairs & valid[:, None] & ~np.isnan(scores))
                    attempts[name].append((scores[query, target], signs[query]))
            attempts = {name: tuple(np.concatenate(parts) for parts in zip(*draws)) for name, draws in attempts.items()}
            threshold = np.quantile(attempts["correct"][0], args.rejected)
            rows.append({"run": run, "k": k, "threshold": round(float(threshold), 3)} | {
                name: accept_rate(score >= threshold, attempted, len(sign_names), rng)
                for name, (score, attempted) in attempts.items() if name != "correct"
            })  # fmt: skip
    with pl.Config(tbl_width_chars=160, fmt_str_lengths=30):
        print(pl.DataFrame(rows).sort("k", maintain_order=True))


if __name__ == "__main__":
    main()
