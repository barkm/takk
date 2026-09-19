"""Web app for practicing signs: pick a sign, sign it to the webcam, and learn whether it was that sign.

The landmarks are extracted in the browser and only they are sent (see practice.py). An attempt
counts as the sign when its mean cosine similarity to the sign's glossary clips reaches --threshold.
The default, 0.38, is the equal-error point of the Swedish recordings against the whole lexicon with
the default run (see the decisions in ROADMAP.md): about 2% of correct attempts are rejected and 2%
of random wrong signs accepted; similar signs get through more often.

Run from the repo root, then open http://localhost:8002 (the browser only gives access to the
camera on localhost or over https, so forward the port when the machine is remote):
    uv run scripts/practice.py
Embedding the glossary takes about a minute on the CPU. The embeddings are cached next to the run
(outputs/runs/<run>/embeddings_<glossary>.npy) and computed again only when the run's model or the
glossary is newer than the cache.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn

from isolated_sign_validation.collection import video_paths
from isolated_sign_validation.dataset import SignDataset
from isolated_sign_validation.extraction import download_model
from isolated_sign_validation.practice import create_app, sign_means
from isolated_sign_validation.preparation import PrepConfig, PreparedData
from isolated_sign_validation.speech import MODEL as SPEECH_MODEL
from isolated_sign_validation.speech import load_aligner
from isolated_sign_validation.training import embed, load_run

RUNS_DIR = Path("outputs/runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared evaluation set whose test signs are practiced")  # fmt: skip
    parser.add_argument("--run", default="iv14_h384_e20", help="training run whose model scores the attempts")
    parser.add_argument("--threshold", type=float, default=0.38, help="lowest score that counts as the sign")
    parser.add_argument("--device", default="cpu", help="the GPU is shared; one attempt at a time is cheap on the CPU")
    parser.add_argument("--no-speech", action="store_true", help="split a sentence at the rests between the signs instead of by the spoken words, and skip the speech model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    download_model()
    glossary = PreparedData(args.glossary)
    clips = SignDataset(glossary, "test")
    _, model = load_run(RUNS_DIR / args.run, glossary)
    model = model.to(args.device)
    cache = RUNS_DIR / args.run / f"embeddings_{args.glossary.name}.npy"
    sources = [RUNS_DIR / args.run / "best.pt", args.glossary / "frames.npy", args.glossary / "clips.parquet"]
    if cache.exists() and cache.stat().st_mtime > max(path.stat().st_mtime for path in sources):
        embeddings = np.load(cache)
    else:
        print(f"embedding {len(clips)} clips of {len(clips.signs)} signs from {args.glossary.name} with {args.run}")
        embeddings = embed(model, clips, args.device)
        np.save(cache, embeddings)
    means = sign_means(embeddings, clips.labels, len(clips.signs))
    table = glossary.clips[clips.positions].with_columns(label=clips.labels)
    references = dict(table.group_by("label").agg("sign", "clip_id").sort("label").select(pl.col("sign").list.first(), "clip_id").iter_rows())
    aligner = None
    if not args.no_speech:
        print(f"loading the Swedish speech model {SPEECH_MODEL} that times a spoken sentence's words (about 1.2 GB, downloaded once)")
        aligner = load_aligner(args.device)
    app = create_app(references, means, video_paths(table), model, PrepConfig(), args.threshold, args.device, aligner)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
