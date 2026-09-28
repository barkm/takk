"""Build the bundle the practice app serves, from a published verifier and a prepared glossary.

The model comes as a verifier (`scripts/export_verifier.py`): by default the one pinned in
models/verifier.txt, downloaded from $ISV_VERIFIER_URI once and kept in outputs/verifiers/, or with
`--verifier` a saved one on disk, to try an export before it is published. The bundle holds that
verifier, the glossary's signs as its references, the lexicon addresses their clips are watched at
and the words that lead to them, and nothing of the data pipeline. The API reads only this, so an
image can be built without a dataset, a prepared store or a training run. Embedding the glossary's
clips is most of the time it takes, and shows its progress.

Run from the repo root:
    ISV_VERIFIER_URI=gs://<bucket>/<prefix> uv run scripts/build_serving.py --glossary data/prepared/sts_lexikon-234f4575
Writes outputs/serving/<run>-<glossary>/.
"""

import argparse
import os
from pathlib import Path

import polars as pl

from isolated_sign_verification.verifier import Verifier, fetch, publish
from sign_data.datasets.sts_lexikon import video_urls
from takk import bundle
from takk.vocabulary import search_index, sign_forms

VERIFIER_PIN = Path("models/verifier.txt")  # the verifier the apps build from, see scripts/export_verifier.py
VERIFIERS_DIR = Path("outputs/verifiers")
SERVING_DIR = Path("outputs/serving")
PIN_FILE = Path("deploy/bundle.txt")  # which bundle the deployed API serves; committed, unlike the bucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verifier", type=Path, help=f"a saved verifier to use instead of the one pinned in {VERIFIER_PIN}")  # fmt: skip
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared evaluation set whose test signs are practiced")  # fmt: skip
    parser.add_argument("--device", default="cpu", help="the GPU is shared, so the glossary is embedded on the CPU")
    parser.add_argument("--out", type=Path, help="where to write the bundle (default outputs/serving/<run>-<glossary>)")  # fmt: skip
    parser.add_argument("--publish", action="store_true", help=f"also upload it to $TAKK_BUNDLE_URI and pin it in {PIN_FILE}")  # fmt: skip
    args = parser.parse_args()
    uri = os.environ.get("TAKK_BUNDLE_URI", "")
    if args.publish and not uri:
        parser.error("--publish needs TAKK_BUNDLE_URI, the gs:// prefix the bundles are kept under")
    if not args.verifier and not os.environ.get("ISV_VERIFIER_URI"):
        parser.error(f"the verifier pinned in {VERIFIER_PIN} is downloaded from ISV_VERIFIER_URI; set it or pass --verifier")

    verifier_dir = args.verifier or fetch(VERIFIER_PIN, os.environ["ISV_VERIFIER_URI"], VERIFIERS_DIR)
    verifier = Verifier.load(verifier_dir, args.device)
    references = verifier.references(args.glossary)

    table = pl.read_parquet(args.glossary / "clips.parquet").filter(pl.col("split") == "test")
    urls = video_urls()
    clip_urls = {
        sign: [urls[clip] for clip in ids if clip in urls]
        for sign, ids in table.group_by("sign").agg("clip_id").iter_rows()
    }
    out = args.out or SERVING_DIR / f"{verifier.meta['run']}-{args.glossary.name}"
    source = {"glossary": args.glossary.name, "verifier": verifier_dir.name}
    bundle.write(out, verifier, references, clip_urls, search_index(table), sign_forms(), source)
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) / 1e6
    print(f"{out}: {len(references.signs)} signs from verifier {verifier_dir.name} (threshold {verifier.threshold}), {size:.0f} MB")  # fmt: skip
    if args.publish:
        print(f"published {publish(out, uri, PIN_FILE)}, pinned in {PIN_FILE}: commit it to deploy it")


if __name__ == "__main__":
    main()
