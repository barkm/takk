import json

import polars as pl
import pytest

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE
from takk.vocabulary import HISTORICAL, category_words, packs, starter_packs, word_index


def entries(tmp_path, *rows: dict):
    defaults = {"video": "/movies/00/x-tecken.mp4", "word": None, "also": None, "categories": [], "lexicon_hits": None, "corpus_hits": None}  # fmt: skip
    full = [{**defaults, **row} for row in rows]
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
        {"id": "01811", "word": "abborre", "categories": category("Djur > fisk")},
        {"id": "05397", "word": "braxen", "categories": category("Djur > fisk"), "lexicon_hits": 2},  # the same sign
        {"id": "09999", "word": "tapir", "categories": category("Djur > fisk")},  # not in this glossary
        {"id": "02299", "word": "jul", "categories": []},
    )

    # one sign, so one word: the one the lexicon counts more often, not the entry the sign is named for
    assert category_words(CLIPS, raw_dir) == {"Djur": [{"sign": "sts:abborre-01811", "word": "braxen"}]}


def test_a_category_names_a_word_by_its_own_entry_and_never_twice(tmp_path):
    raw_dir = entries(
        tmp_path,
        {"id": "01811", "word": "abborre, aborre", "categories": category("Djur > fisk")},
        {"id": "02299", "word": "abborre", "categories": category("Djur > fisk"), "lexicon_hits": 5},  # another form
    )

    # two forms written the same way would be an ambiguous flash card, so the counted one is kept
    assert category_words(CLIPS, raw_dir) == {"Djur": [{"sign": "sts:jul-02299", "word": "abborre"}]}


def test_an_entry_in_several_categories_is_in_each_of_them(tmp_path):
    raw_dir = entries(tmp_path, {"id": "01811", "word": "abborre", "categories": category("Djur > fisk", "Mat och dryck > fisk")})  # fmt: skip

    # the word is not sent when the sign is named for it, which the page fills in itself
    assert category_words(CLIPS, raw_dir) == {
        "Djur": [{"sign": "sts:abborre-01811"}],
        "Mat och dryck": [{"sign": "sts:abborre-01811"}],
    }
    assert category_words(CLIPS, raw_dir, deep=True) == {
        "Djur > fisk": [{"sign": "sts:abborre-01811"}],
        "Mat och dryck > fisk": [{"sign": "sts:abborre-01811"}],
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
        {"name": "Djur", "kind": "category", "words": [{"sign": "sts:abborre-01811"}]},  # named for its word
    ]


def test_a_category_starts_with_the_words_the_lexicon_counts_most(tmp_path):
    raw_dir = entries(
        tmp_path,
        {"id": "01811", "word": "abborre", "categories": category("Djur")},  # never counted, so last
        {"id": "02299", "word": "jul", "categories": category("Djur"), "lexicon_hits": 3},
        {"id": "01267", "word": "livsmedel", "categories": category("Djur"), "lexicon_hits": 3, "corpus_hits": 9},
    )
    path = tmp_path / "packs.json"
    path.write_text(json.dumps({}, ensure_ascii=False))

    assert packs(CLIPS, raw_dir, path) == [
        {
            "name": "Djur",
            "kind": "category",
            # the corpus breaks the tie between the two counted three times in the lexicon
            "words": [{"sign": "sts:livsmedel-01265"}, {"sign": "sts:jul-02299"}, {"sign": "sts:abborre-01811"}],
        }
    ]
