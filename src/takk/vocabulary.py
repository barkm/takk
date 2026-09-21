"""What a learner can search for and choose to learn (step 9 of ROADMAP-takk.md).

Everything here is built from the crawl of Svenskt teckenspråkslexikon
(`data/raw/sts-lexikon/entries.jsonl`, see the README), so nothing is generated and no LLM is
involved. A learner searches for a word or for a theme and ticks the signs they want; there are no
ready-made lists to turn on, which is what the starter packs and the category picker used to be.

A **word** is the heading of a lexicon entry, or the other wording its page shows under the title.
The word a learner is asked to sign is not always the label of the sign that scores it: signs of one
form are one class labelled by its lowest entry, so "äta" is scored as `sts:livsmedel-01265` while
the learner still reads "äta".

A **theme** is the lexicon's own subject category ("Djur", "Djur > fisk"), which every entry page
names. They are subject areas of a dictionary rather than a learning order — Sport and Geografi are
the largest — which is exactly what makes them a search index and made them a poor beginner's list.
"""

import re
from pathlib import Path
from typing import NamedTuple

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE, RAW_DIR

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


class Index(NamedTuple):
    """What a search reads: every practicable word, the most counted first, and the themes."""

    words: list[dict]
    themes: dict[str, list[dict]]


def search_index(clips: pl.DataFrame, raw_dir: Path = RAW_DIR) -> Index:
    """Everything a learner can search for: each Swedish word the glossary can score a sign for, and
    each of the lexicon's categories as a theme. Österberg's historical forms are in neither.

    A word is ordered by how often the lexicon counts the entry it means, as a category's words are
    (see `category_words`), so a search for a common word does not answer with a rare homograph
    first. Both are built once at startup and only searched afterwards.
    """
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    historical = {row["id"] for row in entries.iter_rows(named=True) if any(c["path"] == HISTORICAL for c in row["categories"])}  # fmt: skip
    hits = dict(zip(entries["id"], (entries["lexicon_hits"].fill_null(0))))
    sign_of = dict(zip(clips["clip_id"], clips["sign"]))
    found = [
        (-hits.get(entry_id, 0), word, _word(sign_of[entry_id], entry_id, word))
        for word, entry_id in word_index(entries.filter(~pl.col("id").is_in(historical))).items()
        if entry_id in sign_of
    ]
    themes = {name: words for name, words in sorted(category_words(clips, raw_dir).items()) if name != HISTORICAL}
    return Index([word for _, _, word in sorted(found, key=lambda each: each[:2])], themes)


def search(index: Index, query: str, limit: int = 30) -> list[dict]:
    """The signs a learner searching for `query` is offered: the words of every theme whose name it
    begins, then the words it begins, then the words it appears in, each sign once.

    A theme comes first because its name is the longer match — searching "mat" means the subject
    rather than the word far more often than the other way round — but the word is never hidden: the
    two are one list, and "mat" itself follows the theme's words.
    """
    wanted = query.strip().lower()
    if not wanted:
        return []
    picked: dict[str, dict] = {}  # by sign, so the better match of a shared form is the one offered
    themed = [words for name, words in index.themes.items() if name.lower().startswith(wanted)]
    for words in themed + [[word for word in index.words if _matches(word, wanted, starts=True)]]:
        for word in words:
            picked.setdefault(word["sign"], word)
    if len(picked) < limit:
        for word in index.words:
            if _matches(word, wanted, starts=False):
                picked.setdefault(word["sign"], word)
    return list(picked.values())[:limit]


def _matches(word: dict, wanted: str, starts: bool) -> bool:
    text = word.get("word") or spoken_word(word["sign"])
    return text.lower().startswith(wanted) if starts else wanted in text.lower()


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
