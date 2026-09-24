"""The API of the practice app: it scores an attempt at a sign, and serves the lexicon. The clips
themselves are watched at the lexicon's own addresses, which it hands the page.

The landmarks are extracted in the browser and only they are sent (see practice.py). An attempt
counts as the sign when its mean cosine similarity to the sign's glossary clips reaches the
threshold. The default, 0.38, is the equal-error point of the Swedish recordings against the whole
lexicon with the default run (see the decisions in ROADMAP.md): about 2% of correct attempts are
rejected and 2% of random wrong signs accepted; similar signs get through more often.

Everything served comes from one bundle (`bundle.py`, built by `scripts/build_serving.py`): the
model, the mean embedding of each sign, the addresses of its clips and the words that lead to them.
No dataset, no prepared store and no training run is read here, which is what lets this run in a
container that holds none of them (step 18 of ROADMAP-takk.md).

The page is a separate site (`frontend/`, see README) that reaches this server through a proxy in
development, so this is one of two processes:
    uv run takk                      # this, on port 8002
    cd frontend && npm run dev       # the page, on port 5173, which is the port to open and forward
The browser only gives access to the camera on localhost or over https, so forward 5173 when the
machine is remote.
"""

import argparse
from pathlib import Path

import anthropic
import uvicorn

from takk import bundle
from takk.practice import create_app
from takk.speech import MODEL as SPEECH_MODEL
from takk.speech import load_aligner
from takk.story import MODEL as STORY_MODEL

BUNDLE_DIR = Path("outputs/serving/iv14_h384_e20-sts_lexikon-234f4575")


def story_writer(model: str) -> anthropic.Anthropic | None:
    """The client that writes the story, or None when there are no credentials for it. The check is
    at startup so that a missing key is a line here rather than a failed story, but it is not fatal:
    without it only new words can be practised."""
    try:
        client = anthropic.Anthropic()  # raises a TypeError of its own when there is no key at all
        client.models.retrieve(model)
    except (anthropic.APIError, TypeError) as error:
        print(f"no stories: {model} is not reachable ({type(error).__name__}), so only new words can be practised")
        return None
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_DIR, help="the serving bundle to serve (scripts/build_serving.py)")  # fmt: skip
    parser.add_argument("--threshold", type=float, help="lowest score that counts as the sign; the bundle's by default")
    parser.add_argument("--device", default="cpu", help="the GPU is shared; one attempt at a time is cheap on the CPU")
    parser.add_argument("--story-model", default=STORY_MODEL, help="the model that writes the story a learner signs their way through")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    served = bundle.load(args.bundle, args.threshold)
    print(f"{args.bundle}: {served.meta['n_signs']} signs of {served.meta['glossary']}, embedded by {served.meta['run']}, threshold {served.threshold}")  # fmt: skip
    model = served.model.to(args.device)
    references = {sign: served.clips[sign] for sign in served.signs}  # in the order of the rows of the means
    print(f"loading the Swedish speech model {SPEECH_MODEL} that times a spoken sentence's words (about 1.2 GB, downloaded once)")
    aligner = load_aligner(args.device)  # a sentence is split by its spoken words, so this is not optional
    app = create_app(references, served.means, model, served.config, served.threshold, args.device, aligner, served.vocabulary, served.forms, story_writer(args.story_model))  # fmt: skip
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
