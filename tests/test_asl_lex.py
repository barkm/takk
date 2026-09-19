import polars as pl

from isolated_sign_validation.datasets.asl_lex import FEATURES, read_phonology, signbank_twins


def test_read_phonology(tmp_path):
    columns = ["EntryID", "Code", *(f"{feature}.2.0" for feature in FEATURES), "HandshapeM2.2.0"]
    rows = [["apple", "A_03_054", *(["x"] * len(FEATURES)), "y"], ["café", "B_01_032", *(["z"] * len(FEATURES)), ""]]
    (tmp_path / "signdata.csv").write_bytes(("\n".join(",".join(row) for row in [columns, *rows]) + "\n").encode("latin1"))

    phonology = read_phonology(tmp_path)

    assert phonology.columns == ["Code", *(f"phonology.{feature}" for feature in FEATURES)]
    assert phonology["Code"].to_list() == ["A_03_054", "B_01_032"]
    assert phonology["phonology.Handshape"].to_list() == ["x", "z"]
    assert phonology.schema["phonology.Handshape"] == pl.String


def test_signbank_twins(tmp_path):
    (tmp_path / "signdata.csv").write_bytes(
        "Code,SignBankAnnotationID\nA1,EQUAL\nA2,EQUAL\nA3,DOG\nA4,DOG\nA5,\nA6,\nA7,CHAIR\n".encode("latin1")
    )
    codes = pl.DataFrame({"sign": ["FAIR", "EQUAL", "DOG1", "DOG2", "SIT", "SEAT", "CHAIR"], "Code": [f"A{i}" for i in range(1, 8)]})

    # one SignBank entry is a pair; numbered variants of one gloss and signs without an entry are not
    assert signbank_twins(tmp_path, codes) == {("EQUAL", "FAIR")}
