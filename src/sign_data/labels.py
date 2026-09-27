"""Sign labels compared across datasets and sign languages."""

import re


def normalize_label(label: str) -> str:
    """A sign label or English word reduced for matching across datasets and sign languages: lowercase
    letters and spaces, without parenthesized notes and sense numbers ("SAIL1" and "sail (boat)" give "sail")."""
    return re.sub(r"[^a-z ]", "", re.sub(r"\d+$", "", re.sub(r"\(.*?\)", "", label).strip().lower())).strip()
