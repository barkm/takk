"""The word sets a learner practises from: the lexicon's own subject categories.

Svenskt teckenspråkslexikon sorts its entries into Swedish subject categories of its own ("Djur",
"Kläder", "Mat och dryck", see https://teckensprakslexikon.su.se/kategori), nested a few levels deep
on the entry pages ("Djur > fisk"). The crawl reads them off every entry page into the `categories`
field of `data/raw/sts-lexikon/entries.jsonl`, so nothing is generated here and no LLM is involved
(see step 6 of ROADMAP-takk.md).

A category's entries are lexicon ids, which are the clip ids of the prepared glossary, so an id maps
to the sign of the clip it was extracted from. Entries sharing a sign form are one sign, so several
ids of a category can be the same sign; it is kept once, in the lexicon's own order.
"""

from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE, RAW_DIR


def sign_categories(clips: pl.DataFrame, raw_dir: Path = RAW_DIR, deep: bool = False) -> dict[str, list[str]]:
    """The signs of each category, by its Swedish name. `clips` are the glossary's prepared clips.

    A category is the first level of the path the lexicon publishes ("Djur"), or the whole path
    ("Djur > fisk") with `deep`, which gives smaller and narrower sets.
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
