import json

import polars as pl
import pytest

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE
from takk.vocabulary import HISTORICAL, packs, sign_categories, starter_packs, word_index


def entries(tmp_path, *rows: dict):
    full = [{"video": "/movies/00/x-tecken.mp4", "word": None, "also": None, "categories": [], **row} for row in rows]
    (tmp_path / ENTRIES_FILE).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in full))
    return tmp_path


def category(*paths: str) -> list[dict]:
    return [{"slug": path.lower().replace(" > ", "-"), "path": path} for path in paths]


CLIPS = pl.DataFrame(
    {
        "clip_id": ["01811", "05397", "02299", "01267"],
        "sign": ["sts:abborre-01811", "sts:abborre-01811", "sts:jul-02299", "sts:livsmedel-01265"],
    }
)


def test_a_word_finds_the_lowest_entry_that_means_it():
    rows = pl.DataFrame(
        {
            "id": ["02299", "01811", "05397"],
            "word": ["jul, julafton", "abborre", "abborre"],
            "also": [None, None, "braxen"],
        }
    )

    assert word_index(rows) == {
        "jul": "02299",
        "julafton": "02299",  # an entry's word can be several meanings of the one sign
        "abborre": "01811",  # the lowest entry wins, so the same word always gives the same sign
        "braxen": "05397",  # the other wording under the title is a way in as well
    }


def test_a_heading_beats_another_entry_s_other_wording():
    rows = pl.DataFrame({"id": ["09611", "19924"], "word": ["hjälpa", "hjälp"], "also": ["hjälp", None]})

    assert word_index(rows)["hjälp"] == "19924"  # the entry the word is the heading of, not the older one


def test_a_pack_never_teaches_a_historical_sign_form(tmp_path):
    raw_dir = entries(
        tmp_path,
        {"id": "01267", "word": "äta", "categories": category(HISTORICAL)},  # Österberg's 1916 dictionary
        {"id": "02299", "word": "äta"},
    )
    path = tmp_path / "packs.json"
    path.write_text(json.dumps({"Första tecknen": ["äta"]}, ensure_ascii=False))

    assert starter_packs(CLIPS, raw_dir, path)["Första tecknen"] == [
        {"word": "äta", "id": "02299", "sign": "sts:jul-02299"}
    ]


def test_a_pack_keeps_the_word_to_show_apart_from_the_sign_that_scores_it(tmp_path):
    raw_dir = entries(tmp_path, {"id": "01267", "word": "äta"}, {"id": "02299", "word": "jul"})
    path = tmp_path / "packs.json"
    path.write_text(json.dumps({"Första tecknen": ["äta", "jul"]}, ensure_ascii=False))

    assert starter_packs(CLIPS, raw_dir, path) == {
        "Första tecknen": [
            {"word": "äta", "id": "01267", "sign": "sts:livsmedel-01265"},  # the sign form is shared
            {"word": "jul", "id": "02299", "sign": "sts:jul-02299"},
        ]
    }


def test_a_pack_word_the_lexicon_cannot_sign_is_an_error(tmp_path):
    raw_dir = entries(tmp_path, {"id": "02299", "word": "jul"}, {"id": "09999", "word": "tapir"})
    path = tmp_path / "packs.json"
    path.write_text(json.dumps({"Första tecknen": ["jul", "tapir", "godnatt"]}, ensure_ascii=False))

    with pytest.raises(KeyError, match="tapir, godnatt"):  # 09999 has no clip in this glossary
        starter_packs(CLIPS, raw_dir, path)


def test_a_category_keeps_a_shared_sign_once_and_skips_entries_without_a_clip(tmp_path):
    raw_dir = entries(
        tmp_path,
        {"id": "01811", "categories": category("Djur > fisk")},
        {"id": "05397", "categories": category("Djur > fisk")},  # the same sign, another entry
        {"id": "09999", "categories": category("Djur > fisk")},  # not in this glossary
        {"id": "02299", "categories": []},
    )

    assert sign_categories(CLIPS, raw_dir) == {"Djur": ["sts:abborre-01811"]}


def test_an_entry_in_several_categories_is_in_each_of_them(tmp_path):
    raw_dir = entries(tmp_path, {"id": "01811", "categories": category("Djur > fisk", "Mat och dryck > fisk")})

    assert sign_categories(CLIPS, raw_dir) == {
        "Djur": ["sts:abborre-01811"],
        "Mat och dryck": ["sts:abborre-01811"],
    }
    assert sign_categories(CLIPS, raw_dir, deep=True) == {
        "Djur > fisk": ["sts:abborre-01811"],
        "Mat och dryck > fisk": ["sts:abborre-01811"],
    }


def test_a_starter_pack_and_a_category_are_the_same_kind_of_thing(tmp_path):
    raw_dir = entries(
        tmp_path,
        {"id": "01267", "word": "äta"},
        {"id": "01811", "word": "abborre", "categories": category("Djur > fisk", HISTORICAL)},
    )
    path = tmp_path / "packs.json"
    path.write_text(json.dumps({"Första tecknen": ["äta"]}, ensure_ascii=False))

    assert packs(CLIPS, raw_dir, path) == [
        # the word to show travels only when the sign is not named for it, as here ("äta" is signed
        # as sts:livsmedel-01265); a category's words are its signs, and Österberg's is not offered
        {"name": "Första tecknen", "kind": "pack", "words": [{"sign": "sts:livsmedel-01265", "word": "äta"}]},
        {"name": "Djur", "kind": "category", "words": [{"sign": "sts:abborre-01811"}]},
    ]
