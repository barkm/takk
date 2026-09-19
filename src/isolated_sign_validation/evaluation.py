"""k-shot sign verification evaluation.

A model is evaluated through a similarity matrix between all clips of a split (higher is more
similar), e.g. cosine similarities of embeddings or negative DTW distances. For each query clip and
each sign of the split, k reference clips of that sign are drawn from k different signers, never
from the query's signer, and the query's score for the sign is its mean similarity to them.

- Verification: the query's score for its own sign is a positive trial, its scores for all other
  signs are negative trials, except for its sign's twins (signs known to be the same sign form),
  which are neither. ROC-AUC and equal error rate (EER) are computed over all trials pooled,
  i.e. for a single global threshold.
- Identification: the share of queries whose own sign scores highest (top-1) or among the five
  highest (top-5) among all signs of the split but its sign's twins.

Metrics are pooled over several random draws of references, with 95% bootstrap confidence intervals
over signs. Scores are binned into per-sign histograms, which makes resampling signs cheap; AUC and
EER are therefore exact up to the bin resolution.
"""

from collections.abc import Collection

import numpy as np
import polars as pl

N_BINS = 2000


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normalized @ normalized.T


def signer_codes(clips: pl.DataFrame) -> np.ndarray:
    """Integer codes of the clips' signers. Clips without a signer id count as one signer, also next to
    clips that have one (a glossary without signer ids and recordings that have them)."""
    return np.unique(clips["signer"].fill_null("").to_numpy(), return_inverse=True)[1]


def sample_references(signers: np.ndarray, query_signer: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of up to k clips (among clips by `signers`) by different signers other than `query_signer`."""
    candidates = rng.permutation(np.flatnonzero(signers != query_signer))
    _, first = np.unique(signers[candidates], return_index=True)  # one clip per signer
    return candidates[np.sort(first)[:k]]


def draw_scores(
    similarity: np.ndarray, signs: np.ndarray, signers: np.ndarray, k: int, rng: np.random.Generator, rows: np.ndarray | None = None
) -> np.ndarray:
    """Scores of the clips at positions `rows` (default: every clip) as queries for every sign, shape
    (len(rows), n_signs); NaN if no references.

    `signs` and `signers` are integer codes per clip. References are drawn once per query signer and
    sign, from all clips, and the same way whatever `rows` is; `rows` only saves the memory of the
    scores nobody asked for, which with a whole dictionary as the glossary is most of them.
    """
    rows = np.arange(len(signs)) if rows is None else rows
    scores = np.full((len(rows), signs.max() + 1), np.nan)
    by_sign = [np.flatnonzero(signs == sign) for sign in range(signs.max() + 1)]
    for signer in np.unique(signers):
        queries = np.flatnonzero(signers[rows] == signer)
        for sign, clips in enumerate(by_sign):
            references = clips[sample_references(signers[clips], signer, k, rng)]
            if len(references) and len(queries):
                scores[queries, sign] = similarity[np.ix_(rows[queries], references)].mean(axis=1)
    return scores


def _metrics(pos_hist: np.ndarray, neg_hist: np.ndarray, top1: float, top5: float, n_queries: float) -> dict[str, float]:
    """Metrics from histograms of positive and negative scores over the same bins, and identification counts."""
    pos, neg = pos_hist / pos_hist.sum(), neg_hist / neg_hist.sum()
    neg_below = np.cumsum(neg) - neg  # share of negatives in lower bins
    auc = float((pos * (neg_below + neg / 2)).sum())  # ties within a bin count half
    false_reject = np.cumsum(pos) - pos  # threshold at the lower edge of each bin
    false_accept = 1 - neg_below
    i = np.argmin(np.abs(false_reject - false_accept))
    eer = float((false_reject[i] + false_accept[i]) / 2)
    return {"auc": auc, "eer": eer, "top1": top1 / n_queries, "top5": top5 / n_queries}


def evaluate(
    similarity: np.ndarray,
    clips: pl.DataFrame,
    ks: tuple[int, ...] = (1, 3, 5),
    n_draws: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 0,
    queries: np.ndarray | None = None,
    twins: Collection[tuple[str, str]] = (),
    keep_boot: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate a similarity matrix between `clips` (with `sign` and `signer` columns).

    `twins` are pairs of signs known to be one sign form, whose scores for each other are not trials.

    With `queries` (a boolean mask over the clips), only those clips are scored as queries; references
    are still drawn from all clips. Returns a summary with one row per k and metric (estimate and
    95% CI) and a per-sign table. With `keep_boot`, the summary also holds each bootstrap sample
    (`boot`). The references and bootstrap samples depend only on the clips and the seed, not on the
    similarities, so two models evaluated on the same clips can be compared paired, sample by sample.
    """
    rng = np.random.default_rng(seed)
    sign_names, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    signers = signer_codes(clips)
    index = {sign: i for i, sign in enumerate(sign_names)}
    is_twin = np.zeros((len(sign_names), len(sign_names)), dtype=bool)
    for a, b in twins:
        if a in index and b in index:
            is_twin[index[a], index[b]] = is_twin[index[b], index[a]] = True
    query_rows = np.arange(len(signs)) if queries is None else np.flatnonzero(queries)
    summary, per_sign = [], []
    for k in ks:
        pos, neg, rank = [], [], []  # per query and draw
        for _ in range(n_draws):
            scores = draw_scores(similarity, signs, signers, k, rng, query_rows)
            own = scores[np.arange(len(query_rows)), signs[query_rows]]
            others = scores.copy()
            others[np.arange(len(query_rows)), signs[query_rows]] = np.nan
            others[is_twin[signs[query_rows]]] = np.nan
            pos.append(own)
            neg.append(others)
            rank.append((others > own[:, None]).sum(axis=1) + (others == own[:, None]).sum(axis=1) / 2)
        pos, neg, rank = np.concatenate(pos), np.concatenate(neg), np.concatenate(rank)
        query_signs = np.tile(signs[query_rows], n_draws)
        valid = ~np.isnan(pos)  # queries whose own sign has references by other signers

        edges = np.unique(np.nanquantile(np.concatenate([pos[valid], neg[valid].ravel()]), np.linspace(0, 1, N_BINS + 1)))
        n_signs = len(sign_names)
        pos_hist, neg_hist = np.zeros((n_signs, len(edges) + 1)), np.zeros((n_signs, len(edges) + 1))
        np.add.at(pos_hist, (query_signs[valid], np.searchsorted(edges, pos[valid])), 1)
        neg_valid = neg[valid]
        rows = np.repeat(query_signs[valid], neg_valid.shape[1])
        finite = ~np.isnan(neg_valid.ravel())
        np.add.at(neg_hist, (rows[finite], np.searchsorted(edges, neg_valid.ravel()[finite])), 1)
        counts = np.zeros((3, n_signs))  # top-1 hits, top-5 hits, queries per sign
        np.add.at(counts, (0, query_signs[valid]), rank[valid] < 1)
        np.add.at(counts, (1, query_signs[valid]), rank[valid] < 5)
        np.add.at(counts, (2, query_signs[valid]), 1)

        def metrics(weights: np.ndarray) -> dict[str, float]:
            return _metrics(weights @ pos_hist, weights @ neg_hist, *(counts @ weights))

        estimate = metrics(np.ones(n_signs))
        # Resample the signs that have queries; with `queries`, the others carry no trials to resample.
        scored = np.flatnonzero(counts[2])
        boot = [metrics(np.bincount(scored[rng.integers(0, len(scored), len(scored))], minlength=n_signs).astype(float)) for _ in range(n_bootstrap)]
        for name, value in estimate.items():
            low, high = np.percentile([b[name] for b in boot], [2.5, 97.5]) if boot else (np.nan, np.nan)
            summary.append({"k": k, "metric": name, "value": value, "ci_low": low, "ci_high": high})
            if keep_boot:
                summary[-1]["boot"] = [b[name] for b in boot]
        for sign in range(n_signs):
            if counts[2, sign]:
                one_hot = np.zeros(n_signs)
                one_hot[sign] = 1
                per_sign.append({"k": k, "sign": sign_names[sign], "queries": int(counts[2, sign] / n_draws)} | metrics(one_hot))
    return pl.DataFrame(summary), pl.DataFrame(per_sign)
