"""The word sets a learner practises from: the lexicon's own subject categories.

Svenskt teckenspråkslexikon sorts its entries into Swedish subject categories of its own ("Djur",
"Kläder", "Mat och dryck", see https://teckensprakslexikon.su.se/kategori), which
`scripts/download_sts_lexikon.py` crawls into `data/raw/sts-lexikon/categories.jsonl`. Nothing is
generated here, and no LLM is involved (see step 6 of ROADMAP-takk.md).

A category lists lexicon ids, which are the clip ids of the prepared glossary, so an id maps to the
sign of the clip it was extracted from. Entries sharing a sign form are one sign, so several ids of a
category can be the same sign; it is kept once, in the lexicon's own order.
"""

from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import CATEGORIES_FILE, RAW_DIR


def sign_categories(clips: pl.DataFrame, raw_dir: Path = RAW_DIR) -> dict[str, list[str]]:
    """The signs of each category, by its Swedish name. `clips` are the glossary's prepared clips."""
    sign_of = dict(zip(clips["clip_id"], clips["sign"]))
    categories = {}
    for row in pl.read_ndjson(raw_dir / CATEGORIES_FILE).iter_rows(named=True):
        signs = list(dict.fromkeys(sign_of[clip_id] for clip_id in row["ids"] if clip_id in sign_of))
        if signs:
            categories[row["name"]] = signs
    return categories
