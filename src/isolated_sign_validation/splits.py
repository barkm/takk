"""Train/val/test splits that hold out both signs and signers.

- Signs are assigned to train, val or test (~80/10/10) by a hash of the sign label. A sign's split
  therefore doesn't depend on which other signs exist, and a held-out sign is held out in every dataset.
- Test signers are held out completely: test clips are test signs performed by test signers.
- Train and val share signers: val clips are val signs performed by non-test signers. Validation
  thus measures generalization to unseen signs but not to unseen signers; only test measures both.

Clips of test signers performing non-test signs, and of other signers performing test signs, belong
to no split.

Datasets of other sign languages are used for training only (assign_training_only), without their
signs that match a held-out sign by label, or for evaluation only (assign_evaluation_only), in which
case they are never trained on and all of their signs are unseen.
"""

import hashlib
import re
from collections.abc import Collection

import polars as pl

VAL_PERCENT = 10
TEST_PERCENT = 10


def sign_split(sign: str) -> str:
    """The split ("train", "val" or "test") a sign belongs to."""
    bucket = int(hashlib.sha256(sign.encode()).hexdigest(), 16) % 100
    if bucket < TEST_PERCENT:
        return "test"
    return "val" if bucket < TEST_PERCENT + VAL_PERCENT else "train"


def assign_splits(clips: pl.DataFrame, test_signers: Collection[str]) -> pl.DataFrame:
    """Add a `split` column to `clips` (a store's clip table): "train", "val", "test", or null."""
    splits = {sign: sign_split(sign) for sign in clips["sign"].unique()}
    sign = pl.col("sign").replace_strict(splits, return_dtype=pl.String)
    test_signer = pl.col("signer").is_in(list(test_signers))
    return clips.with_columns(
        split=pl.when(test_signer & (sign == "test")).then(sign).when(~test_signer & (sign != "test")).then(sign)
    )


def normalize_label(label: str) -> str:
    """A sign label or English word reduced for matching across datasets and sign languages: lowercase
    letters and spaces, without parenthesized notes and sense numbers ("SAIL1" and "sail (boat)" give "sail")."""
    return re.sub(r"[^a-z ]", "", re.sub(r"\d+$", "", re.sub(r"\(.*?\)", "", label).strip().lower())).strip()


def matching_signs(words: dict[str, Collection[str]], labels: Collection[str]) -> set[str]:
    """The signs (keys of `words`, which gives each sign's label and English words) with a word that
    matches one of `labels` after normalize_label."""
    targets = {normalize_label(label) for label in labels}
    return {sign for sign, sign_words in words.items() if {normalize_label(word) for word in sign_words} & targets}


def assign_training_only(clips: pl.DataFrame, excluded_signs: Collection[str]) -> pl.DataFrame:
    """Add a `split` column for a dataset used only for training: "train", or null for `excluded_signs`.

    Used for datasets of other sign languages, whose signs are all new; excluded are those that could
    be lookalikes of held-out signs (see matching_signs).
    """
    return clips.with_columns(split=pl.when(~pl.col("sign").is_in(list(excluded_signs))).then(pl.lit("train")))


def assign_evaluation_only(clips: pl.DataFrame) -> pl.DataFrame:
    """Add a `split` column for a dataset used only for evaluation: "test" for every clip.

    Used for datasets of another sign language that are never trained on (Slovo), so that all of
    their signs are unseen by construction and no sign-level hold-out is needed.
    """
    return clips.with_columns(split=pl.lit("test"))
