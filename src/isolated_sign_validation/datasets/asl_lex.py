"""Phonological features of ASL signs from ASL-LEX 2.0 (https://osf.io/zpha4/, CC BY 4.0).

ASL-LEX codes each sign's phonology (of its first morpheme): handshape, location, movement and more,
and links most entries to an entry of ASL SignBank, which has one entry per sign form.
Expects `Data Files/signdata.csv` downloaded into RAW_DIR (see README.md).
"""

from pathlib import Path

import polars as pl

from isolated_sign_validation.splits import normalize_label

RAW_DIR = Path("data/raw/asl-lex")
# The phonological features coded in ASL-LEX 2.0, without the derived MarkedHandshape.
FEATURES = (
    "Handshape", "SelectedFingers", "Flexion", "FlexionChange", "Spread", "SpreadChange", "ThumbPosition",
    "ThumbContact", "SignType", "Movement", "RepeatedMovement", "MajorLocation", "MinorLocation",
    "SecondMinorLocation", "Contact", "NonDominantHandshape", "UlnarRotation",
)  # fmt: skip


def read_phonology(raw_dir: Path) -> pl.DataFrame:
    """Each sign's ASL-LEX code and phonological features, as columns `phonology.<feature>` (strings)."""
    signs = pl.read_csv(raw_dir / "signdata.csv", encoding="latin1", infer_schema_length=0)
    return signs.select("Code", *(pl.col(f"{feature}.2.0").alias(f"phonology.{feature}") for feature in FEATURES))


def signbank_twins(raw_dir: Path, codes: pl.DataFrame) -> set[tuple[str, str]]:
    """Pairs of signs (sorted within each pair) that ASL-LEX links to the same ASL SignBank entry, i.e.
    one sign form under several glosses (EQUAL/FAIR). `codes` has columns `sign` and `Code` (the ASL-LEX
    code). Numbered variants of one gloss (DOG1/DOG2) are not pairs: ASL-LEX filmed them separately
    because they are signed differently, and links them to the closest SignBank entry."""
    signs = pl.read_csv(raw_dir / "signdata.csv", encoding="latin1", infer_schema_length=0)
    linked = codes.join(signs.select("Code", "SignBankAnnotationID"), on="Code").drop_nulls("SignBankAnnotationID")
    pairs = set()
    for group in linked.group_by("SignBankAnnotationID").agg(pl.col("sign").unique().sort())["sign"]:
        pairs |= {(a, b) for i, a in enumerate(group) for b in group[i + 1 :] if normalize_label(a) != normalize_label(b)}
    return pairs
