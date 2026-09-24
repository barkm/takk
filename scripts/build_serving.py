"""Build the bundle the practice app serves, from a training run and a prepared glossary.

This is where the model stops being a training artifact and becomes a deployable one (step 12 of
ROADMAP.md): the bundle holds the encoder's weights, one mean embedding per sign, the lexicon
addresses its clips are watched at and the words that lead to them, and nothing of the data pipeline.
The API reads only this, so an image can be built without a dataset, a prepared store or a run
directory (step 18 of ROADMAP-takk.md).

The clip embeddings are the ones cached beside the run by `takk/main.py`; they are computed here when
that cache is missing or older than the model or the glossary.

Run from the repo root:
    uv run scripts/build_serving.py --run iv14_h384_e20 --glossary data/prepared/sts_lexikon-234f4575
Writes outputs/serving/<run>-<glossary>/.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl

from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.datasets.sts_lexikon import video_urls
from isolated_sign_validation.preparation import PreparedData
from isolated_sign_validation.training import embed, load_run
from takk import bundle
from takk.practice import sign_means
from takk.vocabulary import search_index, sign_forms

RUNS_DIR = Path("outputs/runs")
SERVING_DIR = Path("outputs/serving")


def clip_embeddings(run_dir: Path, glossary: Path, clips: SignDataset, model, device: str) -> np.ndarray:
    """The embedding of every clip of the glossary, from the cache beside the run when it is newer
    than both the model and the glossary, as `takk/main.py` reads it."""
    cache = run_dir / f"embeddings_{glossary.name}.npy"
    sources = [run_dir / "best.pt", glossary / "frames.npy", glossary / "clips.parquet"]
    if cache.exists() and cache.stat().st_mtime > max(path.stat().st_mtime for path in sources):
        return np.load(cache)
    print(f"embedding {len(clips)} clips of {len(clips.signs)} signs with {run_dir.name}")
    embeddings = embed(model, clips, device)
    np.save(cache, embeddings)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="iv14_h384_e20", help="training run whose model the bundle serves")
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared evaluation set whose test signs are practiced")  # fmt: skip
    parser.add_argument("--threshold", type=float, default=0.38, help="lowest score that counts as the sign (see the decisions in ROADMAP.md)")  # fmt: skip
    parser.add_argument("--device", default="cpu", help="the GPU is shared; embedding the glossary on the CPU takes about a minute")  # fmt: skip
    parser.add_argument("--out", type=Path, help="where to write the bundle (default outputs/serving/<run>-<glossary>)")  # fmt: skip
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    data = PreparedData(args.glossary)
    clips = SignDataset(data, "test")
    train_config, model = load_run(run_dir, data)
    model = model.to(args.device)
    embeddings = clip_embeddings(run_dir, args.glossary, clips, model, args.device)
    means = sign_means(embeddings, clips.labels, len(clips.signs))

    table = data.clips[clips.positions].with_columns(label=clips.labels)
    urls = video_urls()
    clip_urls = {
        sign: [urls[clip] for clip in ids if clip in urls]
        for sign, ids in table.group_by("sign").agg("clip_id").iter_rows()
    }
    out = args.out or SERVING_DIR / f"{args.run}-{args.glossary.name}"
    bundle.write(
        out,
        signs=clips.signs,
        means=means,
        clips=clip_urls,
        vocabulary=search_index(table),
        forms=sign_forms(),
        config=data.config,
        train_config=dataclasses.asdict(train_config),
        threshold=args.threshold,
        state_dict=model.to("cpu").state_dict(),
        source={"run": args.run, "glossary": args.glossary.name},
    )
    size = sum(path.stat().st_size for path in out.iterdir()) / 1e6
    print(f"{out}: {len(clips.signs)} signs, {means.shape[1]} dimensions, {size:.0f} MB")
    print(json.dumps({key: value for key, value in json.loads((out / bundle.META_FILE).read_text()).items() if key != "train_config"}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
