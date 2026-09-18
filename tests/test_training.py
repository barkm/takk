import polars as pl

from isolated_sign_validation.training import phonology_targets, twin_matrix


def test_phonology_targets():
    clips = pl.DataFrame({
        "sign": ["APPLE", "APPLE", "TREE", "auslan:WHALE", "NIGHT"],
        "phonology.Handshape": ["x", "x", "open_b", None, "x"],
        "phonology.Contact": ["1", "1", "0", None, "0"],
    })  # fmt: skip

    targets, n_classes = phonology_targets(clips, ["APPLE", "NIGHT", "TREE", "auslan:WHALE"])

    assert n_classes == [2, 2]
    assert targets.tolist() == [[1, 1], [1, 0], [0, 0], [-1, -1]]  # classes in sorted order, -1 unknown


def test_twin_matrix():
    matrix = twin_matrix({("A", "C"), ("B", "HELD_OUT")}, ["A", "B", "C"])

    assert matrix.tolist() == [[False, False, True], [False, False, False], [True, False, False]]
    assert twin_matrix({("B", "HELD_OUT")}, ["A", "B"]) is None  # no pair among the training signs
