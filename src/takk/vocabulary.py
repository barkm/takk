"""What a learner practises: the starter packs and the lexicon's own subject categories.

Both are built from the crawl of Svenskt teckenspråkslexikon (`data/raw/sts-lexikon/entries.jsonl`,
see the README), so nothing is generated here and no LLM is involved (step 6 of ROADMAP-takk.md).

A **starter pack** is a short list of everyday Swedish words in `packs.json`, the first signs a TAKK
learner needs. The words are resolved to lexicon entries here rather than written as ids, so the
packs stay readable and a lexicon that has moved an entry shows up as a failure instead of a wrong
sign. The word a learner is asked to sign is not always the label of the sign that scores it: signs
of one form are one class labelled by its lowest entry, so "äta" is scored as `sts:livsmedel-01265`
while the learner still reads "äta".

A **category** is the lexicon's own subject category ("Djur", "Djur > fisk"), which every entry page
names. They are subject areas of a dictionary, not a learning order — Sport and Geografi are the
largest — so they are for browsing, not for a beginner's first lesson.
"""

import json
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE, RAW_DIR

PACKS = Path(__file__).parent / "packs.json"
# Entries of Österberg's 1916 dictionary, whose sign forms are historical and not what to teach.
HISTORICAL = "Österberg 1916"


def word_index(entries: pl.DataFrame) -> dict[str, str]:
    """Each Swedish word to the id of the lowest entry that means it.

    An entry's `word` is one or several meanings of one sign ("inte hänga med, inte begripa, ..."),
    and `also` holds the other wording the page shows under the title ("arbetsvetenskap" is also
    "ergonomi"). Both are ways into the same sign, but a word is looked up as a heading first and
    only then as another entry's other wording: "hjälp" is the heading of one entry and the other
    wording of an older one, and the heading is the sign a learner means.
    """
    index: dict[str, str] = {}
    for field in ("word", "also"):
        for row in entries.sort("id").select("id", field).iter_rows(named=True):
            for word in (row[field] or "").split(","):
                if word.strip():
                    index.setdefault(word.strip().lower(), row["id"])
    return index


def starter_packs(clips: pl.DataFrame, raw_dir: Path = RAW_DIR, path: Path = PACKS) -> dict[str, list[dict]]:
    """The packs of `path`, each word as the word to show, its lexicon id and the sign that scores it.

    Raises a KeyError naming the words the lexicon no longer has a practicable sign for, which is how
    a pack is checked against a newer crawl.
    """
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    historical = {row["id"] for row in entries.iter_rows(named=True) if any(c["path"] == HISTORICAL for c in row["categories"])}  # fmt: skip
    index = word_index(entries.filter(~pl.col("id").is_in(historical)))
    sign_of = dict(zip(clips["clip_id"], clips["sign"]))
    packs, missing = {}, []
    for name, words in json.loads(path.read_text()).items():
        found = []
        for word in words:
            entry_id = index.get(word.lower())
            if entry_id is None or entry_id not in sign_of:
                missing.append(word)
            else:
                found.append({"word": word, "id": entry_id, "sign": sign_of[entry_id]})
        packs[name] = found
    if missing:
        raise KeyError(f"no sign in this glossary for {', '.join(missing)}")
    return packs


def sign_categories(clips: pl.DataFrame, raw_dir: Path = RAW_DIR, deep: bool = False) -> dict[str, list[str]]:
    """The signs of each category, by its Swedish name. `clips` are the glossary's prepared clips.

    A category is the first level of the path the lexicon publishes ("Djur"), or the whole path
    ("Djur > fisk") with `deep`, which gives smaller and narrower sets. A category's entries are
    lexicon ids, which are the clip ids of the glossary; entries of one sign form give one sign,
    kept once, in the lexicon's own order.
    """
    sign_of = dict(zip(clips["clip_id"], clips["sign"]))
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("categories").list.len() > 0)
    categories: dict[str, list[str]] = {}
    for row in entries.select("id", "categories").iter_rows(named=True):
        sign = sign_of.get(row["id"])
        if sign is None:  # an entry without a clip in this glossary
            continue
        for category in row["categories"]:
            name = category["path"] if deep else category["path"].split(">")[0].strip()
            signs = categories.setdefault(name, [])
            if sign not in signs:
                signs.append(sign)
    return categories
