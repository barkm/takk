"""Adapter extracting landmarks from the WLASL videos into a landmark store.

WLASL is distributed as links to YouTube and ASL dictionary sites, many of them dead. The videos
used here come from a mirror of the surviving ones (11,880 of the 21,083 instances), the glosses and
signer ids from WLASL's own metadata. Each video is one sign instance, already trimmed. Instances
without a video are skipped. Expects the mirror unpacked in data/raw/wlasl, i.e. WLASL_v0.3.json and
the videos somewhere below it (see README).

Extraction takes about two hours; an interrupted run resumes where it left off when run again. Run
from the repo root:
    uv run python -m sign_data.datasets.wlasl                # all available videos
    uv run python -m sign_data.datasets.wlasl --signs 20     # a sample
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path

import polars as pl

from sign_data.extraction import extract_store
from sign_data.labels import normalize_label

DATASET = "wlasl"
RAW_DIR = Path("data/raw/wlasl")
STORE_DIR = Path("data/processed") / DATASET
METADATA_FILE = "WLASL_v0.3.json"


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The instances the mirror has: columns clip_id (the video id), sign (the gloss), signer, url (the
    source video) and path."""
    paths = {path.stem: path for path in raw_dir.rglob("*.mp4")}
    glosses = json.loads((raw_dir / METADATA_FILE).read_text())
    rows = [
        {
            "clip_id": instance["video_id"],
            "sign": gloss["gloss"],
            "signer": f"{DATASET}:{instance['signer_id']}",
            "url": instance["url"],
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


def sign_labels(wlasl_signs: Collection[str], asl_citizen_signs: Collection[str]) -> dict[str, str]:
    """Each WLASL gloss's sign label, whatever its split: the ASL Citizen gloss it matches, or its own
    label if it matches none. A gloss matching several variants of one gloss gets them all joined by
    "|" (drink -> "DRINK1|DRINK2"), since the label doesn't say which variant was signed."""
    by_label = defaultdict(set)
    for sign in asl_citizen_signs:
        by_label[_canonical(sign)].add(sign)
    # several matches are variants of one gloss, told apart by a number ASL Citizen's label has
    return {gloss: "|".join(sorted(by_label.get(_canonical(gloss)) or {_canonical(gloss)})) for gloss in wlasl_signs}

def twin_pairs(clips: pl.DataFrame) -> set[tuple[str, str]]:
    """Pairs of sign labels (sorted within each pair) that share a source video: WLASL files one
    dictionary video under every word the sign translates to, so such labels are one sign form.
    `clips` has columns `label` (see `sign_labels`) and `url`."""
    pairs = set()
    for labels in clips.group_by("url").agg(pl.col("label").unique().sort())["label"]:
        pairs |= {(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]}
    return pairs



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
    extract_store(DATASET, videos, store_dir)


if __name__ == "__main__":
    main()
