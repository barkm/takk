"""Compare training runs on the validation split, paired against the first run.

Each run's best checkpoint is evaluated on the val clips (5 reference draws, as at the end of
training), and each later run's gain over the first run is given with a 95% CI from the same
bootstrap samples of signs for both runs (the references and bootstrap samples depend only on the
clips and the seed). A gain counts only if its CI excludes 0.

A run written as `a+b` is an ensemble: the mean of the runs' unit-length embeddings. Every run is
scored with the first run's twins, so all of them are scored on the same trials.

Run from the repo root: uv run scripts/compare_val.py gru_wlasl_twins gru_new
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.evaluation import cosine_similarity, evaluate
from isolated_sign_validation.preparation import PreparedData
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")
METRICS = ["top1", "eer", "top5"]


def val_embeddings(run: str) -> tuple[pl.DataFrame, np.ndarray, set]:
    """The val clips (sorted by clip id), one run's embeddings of them and its prepared data's twins."""
    data = PreparedData(*map(Path, json.loads((RUNS_DIR / run / "prepared.json").read_text())))
    val = SignDataset(data, "val")
    _, model = load_run(RUNS_DIR / run, data)
    clips = data.clips[val.positions]
    order = np.argsort(clips["clip_id"].to_numpy())
    return clips[order], embed(model.to("cuda"), val, "cuda")[order], data.twins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="runs to compare; gains are relative to the first")
    args = parser.parse_args()

    summaries, base_clips = {}, None
    for spec in args.runs:
        parts = [val_embeddings(run) for run in spec.split("+")]
        if base_clips is None:
            base_clips, base_twins = parts[0][0], parts[0][2]
        if not all(part[0]["clip_id"].equals(base_clips["clip_id"]) for part in parts):
            raise SystemExit(f"{spec} has other val clips than {args.runs[0]}; a paired comparison needs the same")
        embeddings = np.mean([e / np.linalg.norm(e, axis=1, keepdims=True) for _, e, _ in parts], axis=0)
        summaries[spec], _ = evaluate(cosine_similarity(embeddings), base_clips, twins=base_twins, keep_boot=True)

    base = args.runs[0]
    for spec, summary in summaries.items():
        joined = summary.join(summaries[base], on=["k", "metric"], suffix="_base")
        print(f"\n{spec}" + ("" if spec == base else f" vs {base}: value, gain [95% paired CI]"))
        for k, group in joined.group_by("k", maintain_order=True):
            cells = []
            for metric, value, boot, value_base, boot_base in group.select("metric", "value", "boot", "value_base", "boot_base").iter_rows():
                if metric not in METRICS:
                    continue
                if spec == base:
                    cells.append(f"{metric} {value:.4f}")
                    continue
                low, high = np.percentile(np.array(boot) - np.array(boot_base), [2.5, 97.5])
                cells.append(f"{metric} {value:.4f} {value - value_base:+.4f} [{low:+.4f}, {high:+.4f}]")
            print(f"  k={k[0]}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
