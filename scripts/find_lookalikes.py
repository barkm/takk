"""Find MM-WLAuslan signs that look like held-out ASL Citizen signs, as seen by a trained model.

Embeds all MM-WLAuslan clips and the ASL Citizen val and test clips with a run's model, and compares
sign embeddings (the normalized mean of a sign's clip embeddings). As a reference for how similar the
same sign looks, each held-out ASL sign's clips are split by signer into two halves, and the two halves'
sign embeddings compared (same sign) as well as those of different signs. Auslan signs whose nearest
held-out ASL sign is about as similar as the same sign are candidate lookalikes; the Auslan signs that
match a held-out sign by label (left out of training) show whether the label match finds lookalikes.

Run from the repo root: uv run scripts/find_lookalikes.py --run gru_hide_low
Writes outputs/results/auslan_lookalikes.csv (each Auslan sign with its nearest held-out ASL sign).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.datasets import mm_wlauslan
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.splits import normalize_label
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
PREPARED_DIR = Path("data/prepared")


def sign_embeddings(embeddings: np.ndarray, signs: np.ndarray) -> tuple[list[str], np.ndarray]:
    """The distinct signs and their normalized mean clip embedding."""
    labels = sorted(set(signs))
    means = np.stack([embeddings[signs == sign].mean(axis=0) for sign in labels])
    return labels, means / np.linalg.norm(means, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="gru_hide_low")
    parser.add_argument("--asl", type=Path, default=PREPARED_DIR / f"asl_citizen-{PrepConfig().id()}")
    parser.add_argument("--auslan", type=Path, default=PREPARED_DIR / f"mm_wlauslan-{PrepConfig().id()}")
    parser.add_argument("--top", type=int, default=30, help="number of candidates to print")
    args = parser.parse_args()

    asl, auslan = PreparedData(args.asl), PreparedData(args.auslan)
    trained = set(auslan.clips.filter(pl.col("split") == "train")["sign"])
    auslan.clips = auslan.clips.with_columns(split=pl.lit("all"))  # all clips, also those left out of training
    _, model = load_run(RUNS_DIR / args.run, asl)
    model = model.to("cuda")

    held_out = [SignDataset(asl, split) for split in ("val", "test")]
    asl_clips = pl.concat([dataset.data.clips[dataset.positions] for dataset in held_out])
    asl_embeddings = np.concatenate([embed(model, dataset, "cuda") for dataset in held_out])
    auslan_set = SignDataset(auslan, "all")
    auslan_clips = auslan.clips[auslan_set.positions]
    auslan_signs, auslan_means = sign_embeddings(embed(model, auslan_set, "cuda"), auslan_clips["sign"].to_numpy())
    asl_signs, asl_means = sign_embeddings(asl_embeddings, asl_clips["sign"].to_numpy())

    # reference: the same held-out ASL sign by two halves of its signers, and different signs
    half = asl_clips.select(pl.col("signer").rank("dense").over("sign") % 2).to_series().to_numpy()
    halves = [sign_embeddings(asl_embeddings[half == h], asl_clips["sign"].to_numpy()[half == h]) for h in (0, 1)]
    assert halves[0][0] == halves[1][0] == asl_signs
    between = halves[0][1] @ halves[1][1].T
    same, different = np.diag(between), between[~np.eye(len(asl_signs), dtype=bool)]
    print(f"{len(asl_signs)} held-out ASL signs; sign embedding similarity by signer halves (5th / 50th / 95th percentile):")
    print(f"  same sign:      {np.round(np.percentile(same, [5, 50, 95]), 3)}")
    print(f"  different sign: {np.round(np.percentile(different, [5, 50, 95]), 3)}, 99.9th {np.percentile(different, 99.9):.3f}")

    similarity = auslan_means @ asl_means.T
    words = mm_wlauslan.sign_words(mm_wlauslan.RAW_DIR)
    asl_labels = [normalize_label(sign) for sign in asl_signs]
    label_match = []  # for each Auslan sign, the held-out ASL signs it matches by label
    for sign in auslan_signs:
        sign_labels = {normalize_label(word) for word in words[sign.removeprefix("auslan:")]}
        label_match.append({j for j, label in enumerate(asl_labels) if label in sign_labels})
    matched_pairs = np.array([similarity[i, j] for i, js in enumerate(label_match) for j in js])
    print(f"\n{len(auslan_signs)} Auslan signs vs held-out ASL signs (5th / 50th / 95th percentile):")
    print(f"  pairs matching by label ({len(matched_pairs)}): {np.round(np.percentile(matched_pairs, [5, 50, 95]), 3) if len(matched_pairs) else '-'}")
    print(f"  all pairs:                {np.round(np.percentile(similarity, [5, 50, 95]), 3)}")

    nearest = similarity.argmax(axis=1)
    result = (
        pl.DataFrame({
            "auslan_sign": auslan_signs,
            "in_training": [sign in trained for sign in auslan_signs],
            "label_match": [", ".join(asl_signs[j] for j in sorted(js)) for js in label_match],
            "nearest_asl_sign": [asl_signs[j] for j in nearest],
            "similarity": similarity[np.arange(len(auslan_signs)), nearest],
            "same_sign_percentile": [float((same <= v).mean()) for v in similarity[np.arange(len(auslan_signs)), nearest]],
        })
        .sort("similarity", descending=True)
    )  # fmt: skip
    threshold = np.percentile(same, 5)
    for in_training, group in result.group_by("in_training", maintain_order=True):
        print(f"\nAuslan signs {'in' if in_training[0] else 'left out of'} training: {(group['similarity'] >= threshold).sum()} of "
              f"{group.height} have a nearest held-out ASL sign above the same sign's 5th percentile ({threshold:.3f})")  # fmt: skip
    with pl.Config(tbl_rows=args.top, tbl_width_chars=160, fmt_str_lengths=40):
        print(result.head(args.top))
    out = Path("outputs/results/auslan_lookalikes.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
