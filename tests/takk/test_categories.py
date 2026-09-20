import json

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import CATEGORIES_FILE, parse_category_index, parse_category_page
from takk.categories import sign_categories

# the markup of a /kategori listing, cut down to the parts the crawler reads
INDEX_PAGE = """
<a href="/kategori">&Auml;mne</a>
<a href="/kategori/djur">Djur</a>
<a href="/kategori/mat-och-dryck">Mat och dryck</a>
<a href="/kategori/djur">Djur</a>
"""

LISTING_PAGE = """
<a class="no-underline" href="/ord/01811"><img src="/photos/01/abborre-01811-photo-1-small.jpg" /></a>
<a href="/ord/01811">abborre</a>
<a href="/kategori/djur-fisk">Djur &gt; fisk</a>
<a href="/ord/05397">id</a>
"""


def test_the_index_lists_each_category_once():
    assert parse_category_index(INDEX_PAGE) == [("djur", "Djur"), ("mat-och-dryck", "Mat och dryck")]


def test_a_listing_gives_its_entries_and_an_empty_page_ends_the_category():
    assert parse_category_page(LISTING_PAGE) == ["01811", "05397"]
    assert parse_category_page("<p>inga tecken</p>") == []


def test_a_category_keeps_a_shared_sign_once_and_skips_unknown_ids(tmp_path):
    (tmp_path / CATEGORIES_FILE).write_text(
        json.dumps({"id": "djur", "name": "Djur", "ids": ["01811", "05397", "09999"]}) + "\n"
        + json.dumps({"id": "jul", "name": "Jul", "ids": ["09999"]}) + "\n"
    )  # fmt: skip
    clips = pl.DataFrame({"clip_id": ["01811", "05397", "02299"], "sign": ["sts:abborre-01811", "sts:abborre-01811", "sts:jul-02299"]})  # fmt: skip
    assert sign_categories(clips, tmp_path) == {"Djur": ["sts:abborre-01811"]}
