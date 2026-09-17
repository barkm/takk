import polars as pl

from isolated_sign_validation.datasets.slovo import is_single_sign, read_videos


def test_read_videos(tmp_path):
    pl.DataFrame(
        {
            "attachment_id": ["44e8d2a0", "df5b08f0", "nofc3a5ce"],
            "text": ["Ё", "А", "no_event"],
            "user_id": ["185bd3a8", "9c1f4b2e", "185bd3a8"],
            "train": [True, False, True],
        }
    ).write_csv(tmp_path / "annotations.csv", separator="\t")

    videos = read_videos(tmp_path)

    assert videos["clip_id"].to_list() == ["44e8d2a0", "df5b08f0"]  # the no_event clip is left out
    assert videos["sign"].to_list() == ["Ё", "А"]
    assert videos["signer"].to_list() == ["slovo:185bd3a8", "slovo:9c1f4b2e"]
    assert videos["path"].to_list() == [f"{tmp_path}/train/44e8d2a0.mp4", f"{tmp_path}/test/df5b08f0.mp4"]


def test_is_single_sign():
    assert is_single_sign("дятел")
    assert is_single_sign("пока (что)")  # a parenthesized note is not a second word
    assert is_single_sign("кусок; тонкими слоями")  # one gloss of the synonyms is a single word
    assert not is_single_sign("С днем рождения")
    assert not is_single_sign("тот же самый (одинаковый)")
