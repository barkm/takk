"""Export a training run's model as a verifier, the one form of it an application gets.

The verifier is the run's best model, fed clips prepared as `--prepared` is (the preparation of the
references it will score against), with the threshold an attempt has to reach. It carries no
training data and nothing about a sign language, so one export serves any app and glossary.

With `--publish` it is uploaded to $ISV_VERIFIER_URI as one archive named by its own digest, and the
name is pinned in models/verifier.txt: committing that line is what hands the model to the apps,
which build from the pinned verifier (see `verifier.fetch`).

Run from the repo root:
    uv run scripts/isolated_sign_verification/export_verifier.py --run iv14_h384_e20
    ISV_VERIFIER_URI=gs://<bucket>/<prefix> uv run scripts/isolated_sign_verification/export_verifier.py --run iv14_h384_e20 --publish
Writes outputs/verifiers/<run>-<preparation config id>/.
"""

import argparse
import json
import os
from pathlib import Path

from isolated_sign_verification.verifier import META_FILE, Verifier, publish

RUNS_DIR = Path("outputs/runs")
VERIFIERS_DIR = Path("outputs/verifiers")
PIN_FILE = Path("models/verifier.txt")  # which verifier the apps build from; committed, unlike the bucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="iv14_h384_e20", help="training run whose best model is exported")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared set whose preparation the verifier is fed with")  # fmt: skip
    parser.add_argument("--threshold", type=float, default=0.38, help="lowest score that counts as the sign (see ROADMAP.md)")  # fmt: skip
    parser.add_argument("--publish", action="store_true", help=f"also upload it to $ISV_VERIFIER_URI and pin it in {PIN_FILE}")  # fmt: skip
    args = parser.parse_args()
    uri = os.environ.get("ISV_VERIFIER_URI", "")
    if args.publish and not uri:
        parser.error("--publish needs ISV_VERIFIER_URI, the gs:// prefix the verifiers are kept under")

    verifier = Verifier.from_run(RUNS_DIR / args.run, args.prepared, args.threshold)
    out = VERIFIERS_DIR / f"{args.run}-{verifier.config.id()}"
    verifier.save(out)
    meta = json.loads((out / META_FILE).read_text())
    print(f"{out}: threshold {meta['threshold']}, commit {meta['commit']}, model sha256 {meta['model_sha256'][:12]}")
    if args.publish:
        print(f"published {publish(out, uri, PIN_FILE)}, pinned in {PIN_FILE}: commit it to hand it to the apps")


if __name__ == "__main__":
    main()
