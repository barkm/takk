"""Web app for recording yourself signing, to collect a held-out evaluation set (see collection.py).

It shows reference clips of a held-out ASL Citizen sign, or with --lexicon of a Swedish Sign Language
sign from Svenskt teckenspråkslexikon, and records your attempt with the webcam.
No score is shown: the recordings are only checked for whether they are usable at all.

Run from the repo root, then open http://localhost:8000 (the browser only gives access to the
camera on localhost or over https, so forward the port when the machine is remote):
    uv run scripts/collect.py --signs 25 --takes 3
    uv run scripts/collect.py --lexicon --signs 25 --takes 3  # into data/raw/recordings_sts/
The sign list is fixed by --signs and --seed, so several signers can record the same signs.
Recordings are written to data/raw/recordings/, ready for
`uv run python -m isolated_sign_validation.datasets.recordings`.
"""

import argparse
from pathlib import Path

import uvicorn

from isolated_sign_validation.collection import RAW_DIR, Recordings, create_app, lexicon_prompts, session_prompts
from isolated_sign_validation.datasets import asl_citizen, sts_lexikon
from isolated_sign_validation.extraction import download_model
from isolated_sign_validation.preparation import PrepConfig

LEXICON_RAW_DIR = RAW_DIR.with_name("recordings_sts")  # kept apart from the ASL recordings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signs", type=int, default=25, help="how many held-out signs to record")
    parser.add_argument("--takes", type=int, default=3, help="how many recordings of each sign")
    parser.add_argument("--references", type=int, default=3, help="reference clips shown per ASL Citizen sign, by different signers")
    parser.add_argument("--seed", type=int, default=0, help="chooses the signs and their reference clips")
    parser.add_argument("--lexicon", action="store_true", help="Swedish signs from Svenskt teckenspråkslexikon")
    parser.add_argument("--raw_dir", type=Path, help=f"where the recordings are stored (default {RAW_DIR}, {LEXICON_RAW_DIR} with --lexicon)")
    parser.add_argument("--asl_citizen_dir", type=Path, default=asl_citizen.RAW_DIR)
    parser.add_argument("--lexicon_dir", type=Path, default=sts_lexikon.RAW_DIR)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let others on the network record")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    download_model()
    if args.lexicon:
        prompts, reference_dir = lexicon_prompts(args.lexicon_dir, args.signs, args.takes, args.seed), args.lexicon_dir
    else:
        prompts = session_prompts(args.asl_citizen_dir, args.signs, args.takes, args.references, args.seed)
        reference_dir = args.asl_citizen_dir / "videos"
    recordings = Recordings(args.raw_dir or (LEXICON_RAW_DIR if args.lexicon else RAW_DIR))
    print(f"{len(prompts)} prompts; {recordings.clips.height} recordings already in {recordings.path}")
    app = create_app(prompts, recordings, reference_dir, PrepConfig())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
