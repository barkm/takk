"""Build the bundle the practice app serves, from a training run and a prepared glossary.

This is where the model stops being a training artifact and becomes a deployable one: the bundle
holds the run's model as a verifier, the glossary's signs as its references, the lexicon addresses
their clips are watched at and the words that lead to them, and nothing of the data pipeline. The
API reads only this, so an image can be built without a dataset, a prepared store or a run
directory. Embedding the glossary's clips is most of the time it takes, and shows its progress.

Run from the repo root:
    uv run scripts/build_serving.py --run iv14_h384_e20 --glossary data/prepared/sts_lexikon-234f4575
Writes outputs/serving/<run>-<glossary>/.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import polars as pl

from isolated_sign_verification.verifier import Verifier
from sign_data.datasets.sts_lexikon import video_urls
from takk import bundle
from takk.vocabulary import search_index, sign_forms

RUNS_DIR = Path("outputs/runs")
SERVING_DIR = Path("outputs/serving")
PIN_FILE = Path("deploy/bundle.txt")  # which bundle the deployed API serves; committed, unlike the bucket


def publish(bundle_dir: Path, uri: str, pin: Path = PIN_FILE) -> str:
    """Upload `bundle_dir` as one archive named by its own sha256, and write that name and digest to
    `pin`, which is committed. The bucket is not: it comes from TAKK_BUNDLE_URI here and from a
    substitution in the build, so the repository says which bundle is served and never where it is
    kept.

    The name holds the digest, so an object is never overwritten and an older bundle is still there
    to deploy when a new one turns out worse.
    """
    archive = Path(tempfile.gettempdir()) / f"{bundle_dir.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(bundle_dir.iterdir()):
            tar.add(path, arcname=path.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    name = f"{bundle_dir.name}-{digest[:12]}.tar.gz"
    subprocess.run(["gcloud", "storage", "cp", str(archive), f"{uri.rstrip('/')}/{name}"], check=True)
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(f"{name} sha256:{digest}\n")
    archive.unlink()
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="iv14_h384_e20", help="training run whose model the bundle serves")
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared evaluation set whose test signs are practiced")  # fmt: skip
    parser.add_argument("--threshold", type=float, default=0.38, help="lowest score that counts as the sign")  # fmt: skip
    parser.add_argument("--device", default="cpu", help="the GPU is shared, so the glossary is embedded on the CPU")  # fmt: skip
    parser.add_argument("--out", type=Path, help="where to write the bundle (default outputs/serving/<run>-<glossary>)")  # fmt: skip
    parser.add_argument("--publish", action="store_true", help=f"also upload it to $TAKK_BUNDLE_URI and pin it in {PIN_FILE}")  # fmt: skip
    args = parser.parse_args()
    uri = os.environ.get("TAKK_BUNDLE_URI", "")
    if args.publish and not uri:
        parser.error("--publish needs TAKK_BUNDLE_URI, the gs:// prefix the bundles are kept under")

    verifier = Verifier.from_run(RUNS_DIR / args.run, args.glossary, args.threshold, args.device)
    references = verifier.references(args.glossary)

    table = pl.read_parquet(args.glossary / "clips.parquet").filter(pl.col("split") == "test")
    urls = video_urls()
    clip_urls = {
        sign: [urls[clip] for clip in ids if clip in urls]
        for sign, ids in table.group_by("sign").agg("clip_id").iter_rows()
    }
    out = args.out or SERVING_DIR / f"{args.run}-{args.glossary.name}"
    bundle.write(out, verifier, references, clip_urls, search_index(table), sign_forms(), args.glossary.name)
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) / 1e6
    print(f"{out}: {len(references.signs)} signs, {size:.0f} MB")
    print(json.dumps({key: value for key, value in json.loads((out / bundle.VERIFIER_DIR / "verifier.json").read_text()).items() if key != "train_config"}, indent=2))  # fmt: skip
    if args.publish:
        print(f"published {publish(out, uri)}, pinned in {PIN_FILE}: commit it to deploy it")


if __name__ == "__main__":
    main()
