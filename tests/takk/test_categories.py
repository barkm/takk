import json

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import ENTRIES_FILE
from takk.categories import sign_categories


def entries(tmp_path, *rows: dict):
    (tmp_path / ENTRIES_FILE).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return tmp_path


def category(*paths: str) -> list[dict]:
    return [{"slug": path.lower().replace(" > ", "-"), "path": path} for path in paths]


CLIPS = pl.DataFrame(
    {
        "clip_id": ["01811", "05397", "02299"],
        "sign": ["sts:abborre-01811", "sts:abborre-01811", "sts:jul-02299"],
    }
)


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
