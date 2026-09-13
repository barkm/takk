"""Train/val/test splits that hold out both signs and signers.

- Signs are assigned to train, val or test (~80/10/10) by a hash of the sign label. A sign's split
  therefore doesn't depend on which other signs exist, and a held-out sign is held out in every dataset.
- Test signers are held out completely: test clips are test signs performed by test signers.
- Train and val share signers: val clips are val signs performed by non-test signers. Validation
  thus measures generalization to unseen signs but not to unseen signers; only test measures both.

Clips of test signers performing non-test signs, and of other signers performing test signs, belong
to no split.
"""

import hashlib
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
