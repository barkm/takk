from collections import Counter

import polars as pl

from isolated_sign_verification.splits import assign_splits, assign_training_only, matching_signs, sign_split, training_labels
from sign_data.datasets.wlasl import sign_labels
from sign_data.labels import normalize_label


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


def test_normalize_label():
    assert normalize_label("SAIL1") == normalize_label("sail (boat)") == "sail"
    assert normalize_label("TURN ON (START)") == "turn on"
    assert normalize_label("don't") == "dont"


def test_matching_signs_by_label_or_word():
    words = {"WHALE": ["WHALE", "whale"], "HOLY SPIRIT": ["HOLY SPIRIT", "sun", "radiate"], "BAPTISE": ["BAPTISE"]}
    assert matching_signs(words, ["WHALE1", "SUN", "APPLE"]) == {"WHALE", "HOLY SPIRIT"}


def test_assign_training_only():
    clips = pl.DataFrame({"sign": ["A", "B", "A"]})
    assert assign_training_only(clips, excluded_signs={"B"})["split"].to_list() == ["train", None, "train"]


def test_training_labels():
    # WINE is a val sign, the others are train signs (by the hash split, checked below)
    assert sign_split("WINE") != "train" and sign_split("FATHER") == sign_split("CHICKEN") == "train"
    asl_citizen = ["FATHER", "DRINK1", "DRINK2", "WINE", "POLICEMAN1"]

    mapping = training_labels(sign_labels(["father", "drink", "wine", "chicken", "policeman"], asl_citizen))

    assert mapping["father"] == "FATHER"  # a unique match joins the ASL Citizen class
    assert mapping["policeman"] == "POLICEMAN1"  # matching ignores the sense number
    assert mapping["drink"] is None  # several variants, the label doesn't say which
    assert mapping["wine"] is None  # a held-out sign
    assert mapping["chicken"] == "CHICKEN"  # a new sign the hash split puts in train


def test_training_labels_holds_out_twins_of_held_out_signs():
    assert sign_split("WINE") != "train" and sign_split("FATHER") == sign_split("CHICKEN") == "train"

    assert sign_split("SHORT1") == "train" and sign_split("SHORT2") != "train"
    twins = {("CHICKEN", "WINE"), ("FATHER", "SHORT1|SHORT2")}

    mapping = training_labels(sign_labels(["father", "wine", "chicken", "short"], ["FATHER", "WINE", "SHORT1", "SHORT2"]), twins)

    assert mapping["chicken"] is None  # may show the held-out WINE under another word
    assert mapping["father"] is None  # may show the held-out variant SHORT2
    assert mapping["short"] is None
