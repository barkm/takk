import polars as pl

from isolated_sign_validation.datasets.asl_lex import FEATURES, read_phonology


def test_read_phonology(tmp_path):
    columns = ["EntryID", "Code", *(f"{feature}.2.0" for feature in FEATURES), "HandshapeM2.2.0"]
    rows = [["apple", "A_03_054", *(["x"] * len(FEATURES)), "y"], ["café", "B_01_032", *(["z"] * len(FEATURES)), ""]]
    (tmp_path / "signdata.csv").write_bytes(("\n".join(",".join(row) for row in [columns, *rows]) + "\n").encode("latin1"))

    phonology = read_phonology(tmp_path)

    assert phonology.columns == ["Code", *(f"phonology.{feature}" for feature in FEATURES)]
    assert phonology["Code"].to_list() == ["A_03_054", "B_01_032"]
    assert phonology["phonology.Handshape"].to_list() == ["x", "z"]
    assert phonology.schema["phonology.Handshape"] == pl.String
