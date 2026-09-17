"""Web app for recording yourself signing, to collect a held-out evaluation set (see collection.py).

It shows reference clips of a held-out ASL Citizen sign and records your attempt with the webcam.
No score is shown: the recordings are only checked for whether they are usable at all.

Run from the repo root, then open http://localhost:8000 (the browser only gives access to the
camera on localhost or over https, so forward the port when the machine is remote):
    uv run scripts/collect.py --signs 25 --takes 3
The sign list is fixed by --signs and --seed, so several signers can record the same signs.
Recordings are written to data/raw/recordings/, ready for
`uv run python -m isolated_sign_validation.datasets.recordings`.
"""

import argparse
from pathlib import Path

import uvicorn

from isolated_sign_validation.collection import RAW_DIR, Recordings, create_app, session_prompts
from isolated_sign_validation.datasets import asl_citizen
from isolated_sign_validation.extraction import download_model
from isolated_sign_validation.preparation import PrepConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signs", type=int, default=25, help="how many held-out signs to record")
    parser.add_argument("--takes", type=int, default=3, help="how many recordings of each sign")
    parser.add_argument("--references", type=int, default=3, help="reference clips shown per sign, by different signers")
    parser.add_argument("--seed", type=int, default=0, help="chooses the signs and their reference clips")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="where the recordings are stored")
    parser.add_argument("--asl_citizen_dir", type=Path, default=asl_citizen.RAW_DIR)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let others on the network record")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    download_model()
    prompts = session_prompts(args.asl_citizen_dir, args.signs, args.takes, args.references, args.seed)
    recordings = Recordings(args.raw_dir)
    print(f"{len(prompts)} prompts; {recordings.clips.height} recordings already in {recordings.path}")
    app = create_app(prompts, recordings, args.asl_citizen_dir, PrepConfig())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
