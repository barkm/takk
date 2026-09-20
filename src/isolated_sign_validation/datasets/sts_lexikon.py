"""Adapter extracting landmarks from the Svenskt teckenspråkslexikon clips into a landmark store.

Svenskt teckenspråkslexikon is the Swedish Sign Language dictionary and the goal vocabulary, so it
is an evaluation set only and is never trained on (see ROADMAP.md). It is a dictionary rather than a
dataset: one recording per entry and no signer ids. Every entry with a sign video is extracted.
Entries the lexicon marks as sharing a sign form ("Teckenformen kan också betyda") are one sign, a
class of separate recordings of that form under different Swedish meanings, usually by different
model signers; every other entry is a sign with a single clip. A self-recorded clip can be scored
against any entry, but a trial of the lexicon against itself needs a class.

`signer` is null: the lexicon publishes no signer ids, and the pseudo ids by face clustering are a
separate step (see ROADMAP.md). Until they exist, a trial may draw its reference from the same model
signer as the query.

Expects the crawl in data/raw/sts-lexikon (`scripts/download_sts_lexikon.py`, see README).
Run from the repo root:
    uv run python -m isolated_sign_validation.datasets.sts_lexikon             # every entry
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
CATEGORIES_FILE = "categories.jsonl"  # one line per crawled category, with the lexicon ids in it
# A Swedish sign is a different sign from an ASL sign with the same meaning (prepared labels, recordings)
LABEL_PREFIX = "sts:"


FIELDS = ("word", "also", "video", "form", "transcription", "gloss", "english", "same_form", "categories", "lexicon_hits", "corpus_hits", "corpus_total", "survey_hits")  # fmt: skip


def _count(match: re.Match | None) -> int | None:
    """The first group of `match` as a number, or None if it didn't match."""
    return int(match.group(1)) if match else None


def _text(match: re.Match | None) -> str | None:
    """The first group of `match` as plain text, or None if it didn't match or is the empty dash."""
    if match is None:
        return None
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(1)))).strip()
    return None if text in ("", "-") else text


def parse_entry(page: str | None) -> dict:
    """Everything an `/ord/<id>` page says about the entry, as one row.

    All fields are null for a page that doesn't exist (`page` is None) or holds no sign video: the
    lexicon ids are sparse, and not every id is a published entry. `video` is the path of the sign
    video on the site, `form` the Swedish description of the sign's form, `transcription` its
    notation (identical for entries of one form, so it checks the classes), `gloss` the entry's
    gloss in the STS corpus if it has one, and `same_form` whether the entry shares its form with
    others, in which case `/ord/<id>/kan-aven-betyda` lists them.

    The rest is what the page shows around the sign: `also` the other Swedish wording of the same
    entry under its title ("arbetsvetenskap" is also "ergonomi"), `categories` the lexicon's own
    subject categories as (slug, path) with the path as published ("Djur > fisk"), `english` the
    English translation, and the four hit counts of the
    "Förekomster" section. The page's "Uppdaterat" date is the day the page was rendered, the same on
    every entry, so it is not read. The practice app builds its word sets from `categories` and
    can order them by `corpus_hits` (see ROADMAP-takk.md). The example sentences, which sit in a
    Livewire block mixed with other entries' films, are not read either.
    """
    video = re.search(r'<source src="(/movies/[^"?]+-tecken\.mp4)', page) if page else None
    if page is None or video is None:
        return dict.fromkeys(FIELDS)
    hits = _text(re.search(r"Förekomster</h4>\s*<p>(.*?)</p>", page, re.S)) or ""
    corpus = re.search(r"Korpusmaterial: (\d+) av totalt (\d+)", hits)
    return {
        "word": _text(re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)),
        "also": _text(re.search(r'<div class="font-caeciliae[^"]*">(.*?)</div>', page, re.S)),
        "video": video.group(1),
        "form": _text(re.search(r"Formbeskrivning</h4>\s*<p>(.*?)</p>", page, re.S)),
        "transcription": _text(re.search(r'<div class="font-trans">(.*?)</div>', page, re.S)),
        "gloss": _text(re.search(r"Glosa i STS-korpus:</b>(.*?)<br", page, re.S)),
        "english": _text(re.search(r"<b>English:</b>(.*?)<br", page, re.S)),
        "same_form": bool(re.search(r"/ord/\d+/kan-aven-betyda", page)),
        "categories": parse_categories(page),
        "lexicon_hits": _count(re.search(r"Lexikonet: (\d+)", hits)),
        "corpus_hits": int(corpus.group(1)) if corpus else None,
        "corpus_total": int(corpus.group(2)) if corpus else None,
        "survey_hits": _count(re.search(r"Enkäter: (\d+)", hits)),
    }


def parse_categories(page: str) -> list[dict]:
    """The subject categories of an `/ord/<id>` page, as the slug and the path the page shows.

    The lexicon sorts its entries into Swedish subject categories of its own, nested a few levels
    deep ("Sport > klubbar och föreningar > NHL", https://teckensprakslexikon.su.se/kategori). Many
    entries are in none. The category listings on the site are not the same data: they miss some of
    an entry's categories and hide the deeper levels, so the entry page is the source.
    """
    links = re.findall(r'href="/kategori/([a-z0-9-]+)"[^>]*>([^<]+)<', page)
    paths = {slug: re.sub(r"\s+", " ", html.unescape(path)).strip() for slug, path in links}
    return [{"slug": slug, "path": path} for slug, path in paths.items()]


def parse_group(page: str) -> list[str]:
    """The lexicon ids on an `/ord/<id>/kan-aven-betyda` page, which include the entry's own."""
    return sorted({match for match in re.findall(r'href="/ord/(\d+)"', page)})


def parse_category_index(page: str) -> list[tuple[str, str]]:
    """The categories of the `/kategori` page as (slug, name), in the order the lexicon lists them.

    The lexicon sorts its entries into subject categories of its own ("Djur", "Kläder"), which the
    practice app builds its word sets from (see ROADMAP-takk.md). The index lists the top categories;
    a category page splits further ("Djur > fisk"), which is not crawled.
    """
    links = re.findall(r'href="/kategori/([a-z0-9-]+)"[^>]*>([^<]+)<', page)
    # the link text is the name followed by the number of subcategories ("Geografi\n (9)")
    names = {slug: re.sub(r"\s*\(\d+\)$", "", re.sub(r"\s+", " ", html.unescape(name)).strip()) for slug, name in links}  # fmt: skip
    return list(names.items())


def parse_category_page(page: str) -> list[str]:
    """The lexicon ids listed on one page of a `/kategori/<slug>` listing, empty past the last page."""
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
    """Every entry with a sign video: columns clip_id, sign, signer and path.

    The clips of a sign class are labeled by the class (`sign_classes`); any other entry is a sign of
    its own, labeled the same way by its word and lexicon id ("kärlek-01855").
    """
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    groups = pl.read_ndjson(raw_dir / GROUPS_FILE)["members"].to_list()
    classes = sign_classes(entries, groups)
    return (
        entries.select(
            clip_id=pl.col("id"),
            sign=pl.col("id").replace_strict(classes, default=pl.col("word") + "-" + pl.col("id"), return_dtype=pl.String),
            signer=pl.lit(None, dtype=pl.String),
            path=pl.lit(f"{raw_dir}/") + pl.col("video").str.strip_prefix("/"),
        )
        .sort("sign", "clip_id")
    )


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
