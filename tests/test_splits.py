from collections import Counter

import polars as pl

from isolated_sign_validation.splits import assign_splits, sign_split


def test_sign_split_is_roughly_80_10_10():
    counts = Counter(sign_split(f"SIGN{i}") for i in range(10_000))
    assert abs(counts["train"] / 10_000 - 0.8) < 0.02
    assert abs(counts["val"] / 10_000 - 0.1) < 0.02
    assert abs(counts["test"] / 10_000 - 0.1) < 0.02


def test_sign_split_is_pinned():
    # Changing how signs are hashed would silently reshuffle all splits.
    assert {s: sign_split(s) for s in ("APPLE", "SENTENCE1", "CHAMP", "GRASS")} == {
        "APPLE": "train",
        "SENTENCE1": "val",
        "CHAMP": "test",
        "GRASS": "test",
    }


def signs_by_split() -> dict[str, str]:
    """One sign of each split."""
    found = {}
    for i in range(1000):
        found.setdefault(sign_split(f"SIGN{i}"), f"SIGN{i}")
    return found


def test_assign_splits_holds_out_signs_and_test_signers():
    signs = signs_by_split()
    clips = pl.DataFrame(
        [
            {"signer": signer, "sign": signs[split]}
            for signer in ("d:train_signer", "d:test_signer")
            for split in ("train", "val", "test")
        ]
    )
    result = assign_splits(clips, test_signers={"d:test_signer"})

    assert result["split"].to_list() == [
        "train",  # training signer, training sign
        "val",  # training signer, validation sign
        None,  # training signer, test sign: held out from everything but test
        None,  # test signer, training sign: test signers never appear outside test
        None,  # test signer, validation sign
        "test",  # test signer, test sign
    ]
