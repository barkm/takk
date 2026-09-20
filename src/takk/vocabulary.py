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

The learner picks what today's practice draws from, and to them a starter pack and a category are
the same thing: a named list of words to turn on or off (`packs`).
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE, RAW_DIR

PACKS = Path(__file__).parent / "packs.json"
# Entries of Österberg's 1916 dictionary, whose sign forms are historical and not what to teach.
HISTORICAL = "Österberg 1916"


def spoken_word(sign: str) -> str:
    """The Swedish word a lexicon sign is signed for, which is what the signer says: its name without
    the source and the entry number ("sts:platta slag-25563" -> "platta slag")."""
    return re.sub(r"-\d+$", "", sign.removeprefix("sts:"))


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


def packs(clips: pl.DataFrame, raw_dir: Path = RAW_DIR, path: Path = PACKS) -> list[dict]:
    """Everything a learner can choose to practise, as one kind of thing: a named list of words, each
    with the sign that scores it (user, 2026-09-20: a pack is either a starter pack or a category,
    and it does not matter which). The starter packs come first, in the order they are written, then the
    lexicon's categories by name, each ordered by `category_words`; Österberg's dictionary is left out.
    """
    starters = [
        {"name": name, "kind": "pack", "words": [_word(entry["sign"], entry["id"], entry["word"]) for entry in words]}
        for name, words in starter_packs(clips, raw_dir, path).items()
    ]
    categories = [
        {"name": name, "kind": "category", "words": words}
        for name, words in sorted(category_words(clips, raw_dir).items())
        if name != HISTORICAL
    ]
    return starters + categories


def category_words(clips: pl.DataFrame, raw_dir: Path = RAW_DIR, deep: bool = False) -> dict[str, list[dict]]:
    """The words of each category, the ones the lexicon counts most often first.

    A word is the heading of an entry in the category, not the name of the sign that scores it: signs
    of one form are one class labelled by its lowest entry, and that entry often belongs to another
    subject entirely, so "Djur" would otherwise start at "hane" and "batteri". Two words of one form
    in one category are one word, the more often counted of the two, and so are two forms written the
    same way ("fotboll" is signed in two ways): a flash card shows the word alone, so a word that
    could be answered with either form would be scored against whichever of them was drawn.

    The order is the entry's `lexicon_hits`, then its `corpus_hits`, then the older entry, so it is
    the same on every start. Only 27% of entries are counted at all, so the tail of a large category keeps the
    lexicon's own order; the head is what a session draws from, and there the counts are what a
    learner meets first ("Sport" with träna, fotboll, ishockey).
    """
    sign_of = dict(zip(clips["clip_id"], clips["sign"]))
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("categories").list.len() > 0)
    categories: dict[str, list[Entry]] = {}
    for row in entries.select("id", "word", "categories", "lexicon_hits", "corpus_hits").iter_rows(named=True):
        sign = sign_of.get(row["id"])
        if sign is None:  # an entry without a clip in this glossary
            continue
        word = (row["word"] or spoken_word(sign)).split(",")[0].strip()
        found = Entry(row["lexicon_hits"] or 0, row["corpus_hits"] or 0, word, row["id"], sign)
        for category in row["categories"]:
            name = category["path"] if deep else category["path"].split(">")[0].strip()
            categories.setdefault(name, []).append(found)
    return {name: _best(found) for name, found in categories.items()}


class Entry(NamedTuple):
    """One entry of a category: how often the lexicon and the corpus count it, the word it is a
    heading for, its own id, and the sign class that scores it."""

    hits: int
    corpus: int
    word: str
    id: str
    sign: str


def _best(found: list[Entry]) -> list[dict]:
    """The words of one category, the most counted first, each sign once and each word once: of two
    entries of one form, or two forms written the same way, the more often counted is kept, and the
    lexicon's older entry breaks a tie."""
    def order(entry: Entry) -> tuple:
        return -entry.hits, -entry.corpus, entry.id

    for key in (lambda entry: entry.sign, lambda entry: entry.word):
        best: dict[str, Entry] = {}
        for entry in sorted(found, key=order):
            best.setdefault(key(entry), entry)
        found = list(best.values())
    return [_word(entry.sign, entry.id, entry.word) for entry in sorted(found, key=order)]


def _word(sign: str, entry_id: str, word: str | None = None) -> dict:
    """A pack's word: the entry it is, the sign that scores it, and the word to show only when it is
    not the sign's own name. Thousands of category words travel to the browser, so what it can derive
    is not sent. The entry is the word's own, not the class label: "grön" is entry 00419 of the class
    `sts:land-00416`, and it is that entry's description of the form a learner is shown."""
    named = {"sign": sign, "id": entry_id}
    return named if word is None or word == spoken_word(sign) else {**named, "word": word}


def sign_forms(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """The lexicon's own description of each entry's sign form, in Swedish. It is the only teaching
    text the lexicon publishes ("Flata handen, framåtriktad och uppåtvänd, förs åt vänster ...") and
    it is on all but 30 of the entries with a video, so the app shows it next to the clip.
    """
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).select("id", "form")
    return {row["id"]: row["form"] for row in entries.iter_rows(named=True) if row["form"]}
