"""Find candidate twin signs in ASL Citizen: pairs of glosses that may be one sign form (see ROADMAP.md).

Three independent signals, merged into one ranked list for inspection:
- WLASL: the two glosses share a WLASL source video (the prepared WLASL data's twins.csv).
- ASL-LEX: one sign's SignBank English translations include the other's gloss (CALIFORNIA lists
  "gold"), or the two entries share an ASL-LEX lemma. Words listed by many entries ("animal",
  "time") say little and are ignored.
- Similarity: each sign's clips are split by signer into two halves, and the gap between the pair's
  cross similarity (A's half with B's other half) and the higher of the two signs' own similarity
  (a sign's halves with each other) is computed with a trained model and with the hand-crafted
  baseline embedding. A gap near zero or above means the two signs look as alike as each is with
  itself; taking the higher one keeps an inconsistent sign from looking like everything's twin.
  The trained model has pushed its training signs apart, twins included, so it is blind to pairs of
  two training signs; the hand-crafted embedding has no such bias but is weaker.

Run from the repo root: uv run scripts/find_twins.py --run gru_auslan_phonpool1
Writes outputs/results/asl_citizen_twin_candidates.csv and outputs/asl_citizen_twins/pairs.md (links
to 4 clips of each sign, by signers who signed both where possible).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.baselines import hand_embedding
from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.datasets import asl_citizen, asl_lex
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.splits import normalize_label, sign_split
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
PREPARED_DIR = Path("data/prepared")
OUT_DIR = Path("outputs/asl_citizen_twins")


def canonical(label: str) -> str:
    return normalize_label(label).replace(" ", "")


def half_means(embeddings: np.ndarray, signs: np.ndarray, half: np.ndarray, labels: list[str]) -> list[np.ndarray]:
    """For each signer half, the signs' normalized mean clip embedding, rows in the order of `labels`."""
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = {sign: i for i, sign in enumerate(labels)}
    rows = np.array([index[s] for s in signs])
    means = []
    for h in (0, 1):
        total = np.zeros((len(labels), embeddings.shape[1]))
        np.add.at(total, rows[half == h], embeddings[half == h])
        means.append(total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12))
    return means


def gaps(means: list[np.ndarray]) -> np.ndarray:
    """(n, n) cross similarity of two signs minus the higher of their own similarities (by signer halves)."""
    cross = (means[0] @ means[1].T + means[1] @ means[0].T) / 2
    own = np.diag(means[0] @ means[1].T)
    return cross - np.maximum(own[:, None], own[None, :])


def asl_lex_pairs(signs: list[str], max_entries: int = 3) -> set[tuple[str, str]]:
    """Pairs whose ASL-LEX entries share a lemma, or where one entry's SignBank translations include
    the other's gloss (ignoring words listed by more than `max_entries` entries, and pairs of numbered
    variants of one gloss, which match trivially)."""
    codes = asl_citizen.read_videos(asl_citizen.RAW_DIR).select(sign="Gloss", Code="ASL-LEX Code").unique()
    lex = pl.read_csv(asl_lex.RAW_DIR / "signdata.csv", encoding="latin1", infer_schema_length=0)
    entries = codes.filter(pl.col("sign").is_in(signs)).join(lex.select("Code", "LemmaID", "SignBankEnglishTranslations"), on="Code")
    by_canonical: dict[str, set[str]] = {}
    for sign in signs:
        by_canonical.setdefault(canonical(sign), set()).add(sign)
    words = {sign: {canonical(w) for w in (t or "").split(",")} for sign, t in entries.select("sign", "SignBankEnglishTranslations").iter_rows()}
    listed_by = pl.Series("word", [w for ws in words.values() for w in ws]).value_counts()
    common = set(listed_by.filter(pl.col("count") > max_entries)["word"])
    pairs = set()
    for sign, ws in words.items():
        for word in ws - common:
            pairs |= {tuple(sorted((sign, other))) for other in by_canonical.get(word, ()) if canonical(other) != canonical(sign)}
    for group in entries.group_by("LemmaID").agg("sign")["sign"]:
        pairs |= {tuple(sorted((a, b))) for a in group for b in group if a < b}
    return pairs


def clip_links(clips: pl.DataFrame, a: str, b: str, n: int = 4) -> tuple[str, str]:
    """Markdown links to n clips of each sign, by signers who signed both first."""
    both = set(clips.filter(pl.col("sign") == a)["signer"]) & set(clips.filter(pl.col("sign") == b)["signer"])
    links = []
    for sign in (a, b):
        rows = (
            clips.filter(pl.col("sign") == sign)
            .with_columns(shared=pl.col("signer").is_in(list(both)))
            .sort("shared", "signer", descending=[True, False])
            .unique("signer", keep="first", maintain_order=True)
            .head(n)
        )
        video = lambda clip_id: f"../../{asl_citizen.RAW_DIR}/videos/{clip_id}.mp4"  # noqa: E731
        links.append(" · ".join(f"[{s.split(':')[1]}]({video(c)})" for c, s in rows.select("clip_id", "signer").iter_rows()))
    return links[0], links[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="gru_auslan_phonpool1")
    parser.add_argument("--asl", type=Path, default=PREPARED_DIR / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--wlasl", type=Path, default=PREPARED_DIR / f"wlasl-{PrepConfig().id()}")
    parser.add_argument("--top", type=int, default=150, help="pairs taken from each similarity signal")
    args = parser.parse_args()

    data = PreparedData(args.asl)
    data.clips = data.clips.with_columns(split=pl.lit("all"))  # every clip, whatever its split
    dataset = SignDataset(data, "all")
    clips = data.clips[dataset.positions]
    signs = clips["sign"].to_numpy()
    labels = dataset.signs
    half = clips.select(pl.col("signer").rank("dense").over("sign") % 2).to_series().to_numpy()

    _, model = load_run(RUNS_DIR / args.run, data)
    model_gap = gaps(half_means(embed(model.to("cuda"), dataset, "cuda"), signs, half, labels))
    hand = np.stack([hand_embedding(data[i], data.config) for i in dataset.positions])
    hand = (hand - hand.mean(axis=0)) / (hand.std(axis=0) + 1e-6)
    hand_gap = gaps(half_means(hand, signs, half, labels))

    index = {sign: i for i, sign in enumerate(labels)}
    wlasl = {(a, b) for a, b in PreparedData(args.wlasl).twins if a in index and b in index}
    lex = asl_lex_pairs(labels)
    upper = np.triu_indices(len(labels), k=1)
    top = set()
    for gap in (model_gap, hand_gap):
        best = np.argsort(-gap[upper])[: args.top]
        top |= {(labels[upper[0][k]], labels[upper[1][k]]) for k in best}

    features = [c for c in clips.columns if c.startswith("phonology.")]
    phonology = {row[0]: row[1:] for row in clips.group_by("sign").agg(pl.col(features).first()).select("sign", *features).iter_rows()}
    rows = []
    for a, b in sorted(wlasl | lex | top):
        i, j = index[a], index[b]
        differs = [f.removeprefix("phonology.") for f, x, y in zip(features, phonology[a], phonology[b]) if x != y]
        rows.append({
            "sign_a": a, "sign_b": b, "split_a": sign_split(a), "split_b": sign_split(b),
            "wlasl": (a, b) in wlasl, "asl_lex": (a, b) in lex,
            "model_gap": model_gap[i, j], "hand_gap": hand_gap[i, j],
            "phonology_differs": ", ".join(differs) if differs else "identical coding",
        })  # fmt: skip
    # ranked by the number of signals (the similarity signals counting in the top 0.1% of all pairs),
    # then by the model's gap
    model_cut, hand_cut = (np.percentile(g[upper], 99.9) for g in (model_gap, hand_gap))
    result = (
        pl.DataFrame(rows)
        .with_columns(signals=pl.col("wlasl").cast(int) + pl.col("asl_lex").cast(int) + (pl.col("model_gap") > model_cut).cast(int) + (pl.col("hand_gap") > hand_cut).cast(int))
        .sort("signals", "model_gap", descending=True)
    )  # fmt: skip
    out = Path("outputs/results/asl_citizen_twin_candidates.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out)
    print(f"{result.height} candidate pairs: {len(wlasl)} from WLASL, {len(lex)} from ASL-LEX, {len(top)} from similarity "
          f"(99.9th percentile of the gap over all pairs: model {model_cut:.3f}, hand {hand_cut:.3f})")  # fmt: skip
    print(result.group_by("signals").len().sort("signals", descending=True))
    print(f"wrote {out}")

    lines = [
        "# ASL Citizen twin candidates",
        "",
        f"Written by `scripts/find_twins.py --run {args.run}`. Ranked by the number of signals, then the model's gap.",
        "Gap: the two signs' similarity minus the higher of their similarities with themselves (by signer halves; >= 0: as alike as one sign).",
        "Clip links are by signer; signers who signed both signs come first.",
        "",
    ]
    for rank, row in enumerate(result.iter_rows(named=True), 1):
        signals = [name for name, on in (("WLASL", row["wlasl"]), ("ASL-LEX", row["asl_lex"])) if on]
        links_a, links_b = clip_links(clips, row["sign_a"], row["sign_b"])
        lines += [
            f"## {rank}. {row['sign_a']} / {row['sign_b']}",
            "",
            f"- splits {row['split_a']} / {row['split_b']}; signals: {', '.join(signals) or '-'}; "
            f"gap model {row['model_gap']:+.3f}, hand {row['hand_gap']:+.3f}; ASL-LEX differs in: {row['phonology_differs']}",
            f"- **{row['sign_a']}**: {links_a}",
            f"- **{row['sign_b']}**: {links_b}",
            "",
        ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pairs.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / 'pairs.md'}")


if __name__ == "__main__":
    main()
