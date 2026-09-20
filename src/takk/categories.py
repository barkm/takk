"""The word sets a learner practises from: a category of everyday Swedish words, each one a lexicon
sign the app can score.

The sets are written once, offline, by `scripts/categorize_signs.py` (see the decisions in
ROADMAP-takk.md) into `categories.json` next to this file, so nothing here calls an LLM.
"""

import re
from collections.abc import Iterable
from pathlib import Path

PATH = Path(__file__).parent / "categories.json"
ENTRY = re.compile(r"-(\d+)$")


def word(sign: str) -> str:
    """The Swedish word a lexicon sign is signed for ("sts:platta slag-25563" -> "platta slag")."""
    return ENTRY.sub("", sign.removeprefix("sts:"))


def sign_by_word(signs: Iterable[str]) -> dict[str, str]:
    """The sign of each lowercased word. Several signs share a word when the lexicon has separate
    entries for its meanings; the lowest entry number wins, as the lexicon lists it first."""
    found: dict[str, str] = {}
    for sign in sorted(signs, key=lambda sign: int(match.group(1)) if (match := ENTRY.search(sign)) else 0):
        found.setdefault(word(sign).lower(), sign)
    return found
