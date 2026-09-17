"""Adapter extracting landmarks from the WLASL videos into a landmark store.

WLASL is distributed as links to YouTube and ASL dictionary sites, many of them dead. The videos
used here come from a mirror of the surviving ones (11,880 of the 21,083 instances, see ROADMAP.md),
the glosses and signer ids from WLASL's own metadata. Each video is one sign instance, already
trimmed. Instances whose video the mirror does not have are skipped. Expects the mirror unpacked in
data/raw/wlasl, i.e. WLASL_v0.3.json and the videos somewhere below it (see README).

Extraction takes about two hours; an interrupted run resumes where it left off when run again. Run
from the repo root:
    uv run python -m isolated_sign_validation.datasets.wlasl                # all available videos
    uv run python -m isolated_sign_validation.datasets.wlasl --signs 20     # a sample
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.extraction import download_model, extract_landmarks, silence_native_logs
from isolated_sign_validation.landmarks import write_store_resumable
from isolated_sign_validation.parallel import parallel_map
from isolated_sign_validation.splits import normalize_label, sign_split

DATASET = "wlasl"
RAW_DIR = Path("data/raw/wlasl")
STORE_DIR = Path("data/processed") / DATASET
# Holistic extraction uses ~1.7 cores per process; more workers than this are slower (see ROADMAP.md).
MAX_WORKERS = 8
METADATA_FILE = "WLASL_v0.3.json"


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The instances the mirror has: columns clip_id (the video id), sign (the gloss), signer and path."""
    paths = {path.stem: path for path in raw_dir.rglob("*.mp4")}
    glosses = json.loads((raw_dir / METADATA_FILE).read_text())
    rows = [
        {
            "clip_id": instance["video_id"],
            "sign": gloss["gloss"],
            "signer": f"{DATASET}:{instance['signer_id']}",
            "path": str(paths[instance["video_id"]]),
        }
        for gloss in glosses
        for instance in gloss["instances"]
        if instance["video_id"] in paths
    ]
    return pl.DataFrame(rows)


def _canonical(label: str) -> str:
    """A label reduced for matching, in ASL Citizen's spelling ("thank you" and "THANKYOU1" give "THANKYOU")."""
    return normalize_label(label).replace(" ", "").upper()


def map_signs(wlasl_signs: Collection[str], asl_citizen_signs: Collection[str]) -> dict[str, str | None]:
    """Each WLASL gloss's sign label for training, or None if the gloss is not used (see ROADMAP.md).

    WLASL is ASL, so a gloss that matches exactly one ASL Citizen gloss becomes that gloss and both
    datasets' clips share the class. A gloss matching several variants of one gloss (drink ->
    DRINK1/DRINK2) is None, since the label doesn't say which variant was signed. A gloss whose sign
    is held out, and a new gloss the hash split doesn't put in train, are None as well.
    """
    by_label = defaultdict(set)
    for sign in asl_citizen_signs:
        by_label[_canonical(sign)].add(sign)
    mapping: dict[str, str | None] = {}
    for gloss in wlasl_signs:
        matched = by_label.get(_canonical(gloss), set())
        if len(matched) > 1:  # variants of one gloss, told apart by a number ASL Citizen's label has
            mapping[gloss] = None
            continue
        sign = next(iter(matched)) if matched else _canonical(gloss)
        mapping[gloss] = sign if sign_split(sign) == "train" else None
    return mapping


def extract_clips(videos: Sequence[dict]) -> Iterator[tuple[dict, np.ndarray]]:
    """Extract (metadata, landmarks) for `videos` (rows of read_videos), in order."""
    results = parallel_map(extract_landmarks, [Path(video["path"]) for video in videos], max_workers=MAX_WORKERS, initializer=silence_native_logs)  # fmt: skip
    for video, (landmarks, info) in zip(videos, results):
        metadata = {
            "dataset": DATASET,
            "clip_id": video["clip_id"],
            "sign": video["sign"],
            "signer": video["signer"],
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        }
        yield metadata, landmarks


def convert(videos: pl.DataFrame, store_dir: Path) -> None:
    """Extract landmarks for `videos` into a landmark store, resuming an interrupted run."""
    download_model()
    write_store_resumable(
        store_dir,
        videos.to_dicts(),
        lambda batch: tqdm(extract_clips(batch), total=len(batch), desc=DATASET, unit="clip"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signs", type=int, help="only the videos of this many randomly chosen signs")
    args = parser.parse_args()

    videos, store_dir = read_videos(RAW_DIR), STORE_DIR
    if args.signs:
        signs = videos["sign"].unique().sort().sample(args.signs, seed=0).sort().to_list()
        videos = videos.filter(pl.col("sign").is_in(signs))
        store_dir = STORE_DIR.with_name(f"{DATASET}_{args.signs}_signs")  # a sample goes into a separate store
    print(f"{videos.height} videos of {videos['sign'].n_unique()} signs -> {store_dir}")
    convert(videos, store_dir)


if __name__ == "__main__":
    main()
