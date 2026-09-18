import numpy as np
import polars as pl
import pytest

from isolated_sign_validation.evaluation import _metrics, cosine_similarity, draw_scores, evaluate, sample_references


def test_metrics_from_histograms():
    # positives in bins 1 and 3, negatives in bins 0 and 2: 3 of 4 pairs ordered correctly
    m = _metrics(np.array([0, 1, 0, 1.0]), np.array([1, 0, 1, 0.0]), top1=1, top5=2, n_queries=2)
    assert m == pytest.approx({"auc": 0.75, "eer": 0.5, "top1": 0.5, "top5": 1.0})
    perfect = _metrics(np.array([0, 0, 0, 2.0]), np.array([3, 0, 0, 0.0]), 2, 2, 2)
    assert (perfect["auc"], perfect["eer"]) == (1.0, 0.0)
    same = _metrics(np.array([0, 5, 0.0]), np.array([0, 5, 0.0]), 0, 0, 1)
    assert same["auc"] == 0.5


def test_sample_references_uses_different_signers_other_than_the_query():
    signers = np.array([0, 0, 1, 2, 3, 3, 4])
    rng = np.random.default_rng(0)
    for _ in range(20):
        refs = sample_references(signers, query_signer=3, k=3, rng=rng)
        assert len(refs) == 3 and 3 not in signers[refs] and len(set(signers[refs])) == 3
    assert sorted(signers[sample_references(signers, 3, 10, rng)]) == [0, 1, 2, 4]


def clips_table(n_signs: int, n_signers: int) -> pl.DataFrame:
    return pl.DataFrame([{"sign": f"S{s}", "signer": f"P{p}"} for s in range(n_signs) for p in range(n_signers)])


def test_references_never_include_the_query_signer():
    clips = clips_table(5, 6)
    _, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    _, signers = np.unique(clips["signer"].to_numpy(), return_inverse=True)
    same_signer = (signers[:, None] == signers[None, :]).astype(float)  # similar only within a signer
    scores = draw_scores(same_signer, signs, signers, k=3, rng=np.random.default_rng(0))
    assert (scores == 0).all()


def test_query_subset_scores_only_those_queries():
    clips = clips_table(6, 5)
    _, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    queries = (clips["signer"] == "P0").to_numpy()
    similarity = cosine_similarity(np.eye(6)[signs] + 0.01 * np.random.default_rng(0).normal(size=(len(signs), 6)))
    similarity[queries] = np.random.default_rng(1).normal(size=(queries.sum(), len(signs)))  # P0's queries are noise
    subset, per_sign = evaluate(similarity, clips, ks=(1,), n_bootstrap=0, queries=queries)
    rest, _ = evaluate(similarity, clips, ks=(1,), n_bootstrap=0, queries=~queries)
    top1 = lambda summary: summary.filter(pl.col("metric") == "top1")["value"][0]  # noqa: E731
    assert top1(rest) == 1.0 and top1(subset) < 0.6
    assert (per_sign["queries"] == 1).all()  # one query per sign: P0's clip


def test_perfect_and_random_similarities():
    clips = clips_table(10, 8)
    rng = np.random.default_rng(0)
    _, signs = np.unique(clips["sign"].to_numpy(), return_inverse=True)
    perfect = np.eye(10)[signs] + 0.01 * rng.normal(size=(len(signs), 10))
    summary, per_sign = evaluate(cosine_similarity(perfect), clips, n_bootstrap=100)
    values = dict(zip(zip(summary["k"], summary["metric"]), summary["value"]))
    for k in (1, 3, 5):
        assert values[(k, "auc")] > 0.999 and values[(k, "eer")] < 0.01 and values[(k, "top1")] == 1.0
    assert per_sign.height == 3 * 10 and (per_sign["queries"] == 8).all()

    summary, _ = evaluate(cosine_similarity(rng.normal(size=(len(signs), 64))), clips, n_bootstrap=100)
    for row in summary.filter(pl.col("metric") == "auc").iter_rows(named=True):
        assert 0.4 < row["value"] < 0.6 and row["ci_low"] <= row["value"] <= row["ci_high"]


def test_evaluate_with_a_query_mask_bootstraps_over_the_scored_signs():
    """With a query mask the other signs carry no trials, so resampling them would give empty draws."""
    rng = np.random.default_rng(0)
    n_signs, n_signers = 8, 4
    clips = pl.DataFrame(
        {
            "sign": [f"S{sign}" for sign in range(n_signs) for _ in range(n_signers)],
            "signer": [f"P{signer}" for _ in range(n_signs) for signer in range(n_signers)],
        }
    )
    embeddings = rng.normal(size=(n_signs * n_signers, 16))
    queries = np.zeros(len(clips), dtype=bool)
    queries[clips["sign"].is_in(["S0", "S1"]).to_numpy()] = True

    summary, per_sign = evaluate(cosine_similarity(embeddings), clips, ks=(1,), n_draws=1, n_bootstrap=20, queries=queries)

    assert per_sign["sign"].to_list() == ["S0", "S1"]  # only the scored signs
    assert summary["ci_low"].is_finite().all() and summary["ci_high"].is_finite().all()
