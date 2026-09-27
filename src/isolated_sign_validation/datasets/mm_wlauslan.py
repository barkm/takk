"""Adapter extracting landmarks from the MM-WLAuslan videos into a landmark store.

Only the RGB videos of the front Kinect camera are used, from the subsets in SUBSETS (see ROADMAP.md).
The dataset has no signer ids, so `signer` is null. Expects the downloaded zips unzipped in place,
e.g. data/raw/mm-wlauslan/Valid/Kinect_F/rgb/<sample id>_kf_rgb.mp4.

Extraction of all subsets takes about 9 hours; an interrupted run resumes where it left off when
run again. Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.mm_wlauslan                          # all subsets
    uv run python -m isolated_sign_validation.datasets.mm_wlauslan --subsets Valid --signs 20  # a sample
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from isolated_sign_validation.extraction import extract_store

DATASET = "mm_wlauslan"
RAW_DIR = Path("data/raw/mm-wlauslan")
STORE_DIR = Path("data/processed") / DATASET
CAMERA, CAMERA_CODE = "Kinect_F", "kf"
# Subsets (named as their label files) with intact signing: the studio recordings, and test sets with
# replaced backgrounds; not Test_TED, which removes frames and changes the speed.
SUBSETS = ("Train", "Valid", "Test_STU", "Test_ITW", "Test_SYN")


def read_videos(raw_dir: Path, subsets: Sequence[str] = SUBSETS) -> pl.DataFrame:
    """The videos of `subsets`: columns subset, clip_id (the sample id), sign and path."""
    rows = []
    for subset in subsets:
        labels = json.loads((raw_dir / "Annotation" / "Labels & Split" / f"{subset}.json").read_text())
        folder = raw_dir / subset.replace("_", "-") / CAMERA / "rgb"
        rows += [
            {"subset": subset, "clip_id": sample, "sign": sign, "path": str(folder / f"{sample}_{CAMERA_CODE}_rgb.mp4")}
            for sample, sign in labels.items()
        ]
    return pl.DataFrame(rows)


def sign_words(raw_dir: Path) -> dict[str, list[str]]:
    """Each sign's gloss and English keywords, from the dataset's dictionary."""
    dictionary = json.loads((raw_dir / "WWW_CV_ISLR_Challenge" / "Dictionary_Mapping" / "Dictionary.json").read_text())
    return {sign: [sign, *entry["Keywords"].split(",")] for sign, entry in dictionary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS), help="only these subsets")
    parser.add_argument("--signs", type=int, help="only the videos of this many randomly chosen signs")
    args = parser.parse_args()

    videos, store_dir = read_videos(RAW_DIR, args.subsets), STORE_DIR
    if args.signs:
        signs = videos["sign"].unique().sort().sample(args.signs, seed=0).sort().to_list()
        videos = videos.filter(pl.col("sign").is_in(signs))
    if args.signs or args.subsets != list(SUBSETS):  # a sample goes into a separate store
        store_dir = STORE_DIR.with_name("_".join([DATASET, *map(str.lower, args.subsets), *([f"{args.signs}_signs"] if args.signs else [])]))
    # the extractor would silently turn a missing video into an empty clip
    missing = [path for path in videos["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[0]}; download and unzip them first")
    print(f"{videos.height} videos -> {store_dir}")
    extract_store(DATASET, videos, store_dir)


if __name__ == "__main__":
    main()
