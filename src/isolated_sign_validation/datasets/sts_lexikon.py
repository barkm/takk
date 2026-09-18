"""Adapter extracting landmarks from the Svenskt teckenspråkslexikon clips into a landmark store.

Svenskt teckenspråkslexikon is the Swedish Sign Language dictionary and the goal vocabulary, so it
is an evaluation set only and is never trained on (see ROADMAP.md). It is a dictionary rather than a
dataset: one recording per entry and no signer ids. A sign class therefore comes from the entries
the lexicon marks as sharing a sign form ("Teckenformen kan också betyda"), which are separate
recordings of one form under different Swedish meanings, usually by different model signers. Only
classes with at least two clips are kept, since a class needs two clips for a verification trial;
the entries outside them are still part of the crawl and can serve as negatives.

`signer` is null: the lexicon publishes no signer ids, and the pseudo ids by face clustering are a
separate step (see ROADMAP.md). Until they exist, a trial may draw its reference from the same model
signer as the query.

Expects the crawl in data/raw/sts-lexikon (`scripts/download_sts_lexikon.py`, see README).
Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.sts_lexikon             # all clips of the classes
    uv run python -m isolated_sign_validation.datasets.sts_lexikon --signs 20  # a sample
"""

import argparse
import html
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from isolated_sign_validation.extraction import download_model, extract_landmarks, silence_native_logs
from isolated_sign_validation.landmarks import write_store_resumable
from isolated_sign_validation.parallel import parallel_map

DATASET = "sts_lexikon"
RAW_DIR = Path("data/raw/sts-lexikon")
STORE_DIR = Path("data/processed") / DATASET
# Holistic extraction uses ~1.7 cores per process; more workers than this are slower (see ROADMAP.md).
MAX_WORKERS = 8
BASE_URL = "https://teckensprakslexikon.su.se"
ENTRIES_FILE = "entries.jsonl"  # one line per crawled lexicon id, written by scripts/download_sts_lexikon.py
GROUPS_FILE = "groups.jsonl"  # one line per crawled same-form group
# A Swedish sign is a different sign from an ASL sign with the same meaning (prepared labels, recordings)
LABEL_PREFIX = "sts:"


def _text(match: re.Match | None) -> str | None:
    """The first group of `match` as plain text, or None if it didn't match or is the empty dash."""
    if match is None:
        return None
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(1)))).strip()
    return None if text in ("", "-") else text


def parse_entry(page: str | None) -> dict:
    """The fields of an `/ord/<id>` page: word, video, form, transcription, gloss and same_form.

    All fields are null for a page that doesn't exist (`page` is None) or holds no sign video: the
    lexicon ids are sparse, and not every id is a published entry. `video` is the path of the sign
    video on the site, `form` the Swedish description of the sign's form, `transcription` its
    notation (identical for entries of one form, so it checks the classes), `gloss` the entry's
    gloss in the STS corpus if it has one, and `same_form` whether the entry shares its form with
    others, in which case `/ord/<id>/kan-aven-betyda` lists them.
    """
    video = re.search(r'<source src="(/movies/[^"?]+-tecken\.mp4)', page) if page else None
    if page is None or video is None:
        return dict.fromkeys(("word", "video", "form", "transcription", "gloss", "same_form"))
    return {
        "word": _text(re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)),
        "video": video.group(1),
        "form": _text(re.search(r"Formbeskrivning</h4>\s*<p>(.*?)</p>", page, re.S)),
        "transcription": _text(re.search(r'<div class="font-trans">(.*?)</div>', page, re.S)),
        "gloss": _text(re.search(r"Glosa i STS-korpus:</b>(.*?)<br", page, re.S)),
        "same_form": bool(re.search(r"/ord/\d+/kan-aven-betyda", page)),
    }


def parse_group(page: str) -> list[str]:
    """The lexicon ids on an `/ord/<id>/kan-aven-betyda` page, which include the entry's own."""
    return sorted({match for match in re.findall(r'href="/ord/(\d+)"', page)})


def sign_classes(entries: pl.DataFrame, groups: Sequence[Sequence[str]]) -> dict[str, str]:
    """Each entry id's sign class label, for the entries sharing their form with another entry.

    `entries` are the crawled entries that have a video, `groups` the crawled same-form groups.
    Groups are merged transitively, so an entry linked to two entries that are not linked to each
    other still gives one class. A class is labeled by its lowest lexicon id and that entry's word
    ("förälskad-01854"), which stays the same as long as the lexicon keeps its ids. Entries in no
    group, and groups with only one crawled entry, are left out: one clip allows no trial.
    """
    parent = {entry_id: entry_id for entry_id in entries["id"]}

    def root(entry_id: str) -> str:
        while parent[entry_id] != entry_id:
            parent[entry_id] = parent[parent[entry_id]]
            entry_id = parent[entry_id]
        return entry_id

    for group in groups:
        members = [member for member in group if member in parent]
        for member in members[1:]:
            parent[root(member)] = root(members[0])

    members_by_root: dict[str, list[str]] = {}
    for entry_id in parent:
        members_by_root.setdefault(root(entry_id), []).append(entry_id)
    words = dict(zip(entries["id"], entries["word"]))
    return {
        member: f"{words[min(members)]}-{min(members)}"
        for members in members_by_root.values()
        if len(members) > 1
        for member in members
    }


def read_videos(raw_dir: Path) -> pl.DataFrame:
    """The clips of the signs with at least two recordings: columns clip_id, sign, signer and path."""
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    groups = pl.read_ndjson(raw_dir / GROUPS_FILE)["members"].to_list()
    classes = sign_classes(entries, groups)
    return (
        entries.filter(pl.col("id").is_in(list(classes)))
        .select(
            clip_id=pl.col("id"),
            sign=pl.col("id").replace_strict(classes, return_dtype=pl.String),
            signer=pl.lit(None, dtype=pl.String),
            path=pl.lit(f"{raw_dir}/") + pl.col("video").str.strip_prefix("/"),
        )
        .sort("sign", "clip_id")
    )


def shown_clips(clips: pl.DataFrame) -> list[str]:
    """The clip of each sign class that the collection app shows a signer to copy: its lowest lexicon id.

    `clips` needs clip_id and sign columns (rows of read_videos, or prepared clips). A recording is
    never scored against the clip it copied (see `collection.lexicon_prompts`), so these clips are
    left out of the glossary when recordings are evaluated.
    """
    return clips.group_by("sign").agg(pl.col("clip_id").min())["clip_id"].to_list()


def mixed_transcriptions(raw_dir: Path) -> pl.DataFrame:
    """The sign classes whose entries don't all share one transcription, most entries first.

    The lexicon notates every sign's form, so entries of one form should notate the same way. A
    class that doesn't is a same-form group that holds more than one sign, and is worth viewing
    before the benchmark is trusted (see ROADMAP.md).
    """
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    groups = pl.read_ndjson(raw_dir / GROUPS_FILE)["members"].to_list()
    classes = sign_classes(entries, groups)
    return (
        entries.filter(pl.col("id").is_in(list(classes)))
        .with_columns(sign=pl.col("id").replace_strict(classes, return_dtype=pl.String))
        .group_by("sign")
        .agg(clips=pl.len(), transcriptions=pl.col("transcription").n_unique())
        .filter(pl.col("transcriptions") > 1)
        .sort("clips", descending=True)
    )


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
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--signs", type=int, help="only the videos of this many randomly chosen signs")
    args = parser.parse_args()

    videos, store_dir = read_videos(args.raw_dir), STORE_DIR
    if args.signs:
        signs = videos["sign"].unique().sort().sample(args.signs, seed=0).sort().to_list()
        videos = videos.filter(pl.col("sign").is_in(signs))
        store_dir = STORE_DIR.with_name(f"{DATASET}_{args.signs}_signs")  # a sample goes into a separate store
    mixed = mixed_transcriptions(args.raw_dir)
    print(f"{mixed.height} classes hold more than one transcription and may hold more than one sign")
    # the extractor would silently turn a missing video into an empty clip
    missing = [path for path in videos["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[0]}; crawl them first")
    print(f"{videos.height} videos of {videos['sign'].n_unique()} signs -> {store_dir}")
    convert(videos, store_dir)


if __name__ == "__main__":
    main()
