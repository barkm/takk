"""Phonological features of ASL signs from ASL-LEX 2.0 (https://osf.io/zpha4/, CC BY 4.0).

ASL-LEX codes each sign's phonology (of its first morpheme): handshape, location, movement and more.
Expects `Data Files/signdata.csv` downloaded into RAW_DIR (see README.md).
"""

from pathlib import Path

import polars as pl

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
