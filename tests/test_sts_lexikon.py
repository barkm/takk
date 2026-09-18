import json

import polars as pl

from isolated_sign_validation.datasets.sts_lexikon import parse_entry, parse_group, read_videos, shown_clips, sign_classes

# the markup of an /ord/<id> page, cut down to the parts the adapter reads
ENTRY_PAGE = """
<title>kär - Svenskt teckenspr&aring;kslexikon</title>
<h1 class="">k&auml;r</h1>
<video class="js-player mainvideo h-auto w-full">
    <source src="/movies/06/kar-06051-tecken.mp4?v=2026-04-14" type="video/mp4">
</video>
<h4 class="font-thesans-semibold text-lg">Formbeskrivning</h4>
<p>O-handen uppåtriktad och framåtvänd, förändras till sprethanden
    ett par gånger framför pannan</p>
<p>
    <b>Lexikon-ID:</b> 06051<br />
    <b>Glosa i STS-korpus:</b>
        KÄR
    <br />
</p>
<h4 class="font-thesans-semibold text-lg">Transkription</h4>
<div class="font-trans">􌤕􌥆􌤵􌤷</div>
<p>
    <a class="underline" href="/ord/06051/samma-betydelse">Andra tecken med samma betydelse</a><br />
    <a class="underline" href="/ord/06051/kan-aven-betyda">Teckenformen kan ocks&aring; betyda</a>
</p>
"""

GROUP_PAGE = """
<title>kär - Kan även betyda - Svenskt teckenspråkslexikon</title>
<a href="/ord/06051/kan-aven-betyda">Teckenformen kan också betyda</a>
<a href="/ord/01854">förälskad</a>
<a href="/ord/05788">förälskelse</a>
<a href="/ord/06051">kär</a>
"""


def test_parse_entry():
    entry = parse_entry(ENTRY_PAGE)

    assert entry["word"] == "kär"  # the entities are unescaped
    assert entry["video"] == "/movies/06/kar-06051-tecken.mp4"  # without the cache-busting query
    assert entry["form"].startswith("O-handen uppåtriktad")
    assert entry["form"].endswith("framför pannan")  # the description is one line, whatever the markup wraps
    assert entry["transcription"] == "􌤕􌥆􌤵􌤷"
    assert entry["gloss"] == "KÄR"
    assert entry["same_form"]


def test_parse_entry_without_a_sign_video():
    missing = parse_entry(None)  # an id that is not a published entry
    without_video = parse_entry(ENTRY_PAGE.replace("-tecken.mp4", "-fras-1.mp4"))  # only an example sentence

    assert missing == without_video
    assert set(missing) == {"word", "video", "form", "transcription", "gloss", "same_form"}
    assert not any(missing.values())


def test_parse_entry_without_a_gloss():
    page = ENTRY_PAGE.replace("KÄR\n    <br />", "-\n    <br />")  # the lexicon writes a dash for no gloss

    assert parse_entry(page)["gloss"] is None
    assert not parse_entry(ENTRY_PAGE.replace("kan-aven-betyda", "samma-betydelse"))["same_form"]


def test_parse_group():
    assert parse_group(GROUP_PAGE) == ["01854", "05788", "06051"]  # the page's own entry included, the subpage not


def entries(*ids: str) -> pl.DataFrame:
    return pl.DataFrame({"id": list(ids), "word": [f"word{entry_id}" for entry_id in ids]})


def test_sign_classes():
    classes = sign_classes(entries("00001", "00002", "00003"), [["00002", "00003"]])

    assert classes == {"00002": "word00002-00002", "00003": "word00002-00002"}  # the lowest id names the class
    assert "00001" not in classes  # an entry in no group has one clip and no trial


def test_sign_classes_merges_groups_transitively():
    classes = sign_classes(entries("00001", "00002", "00003"), [["00001", "00002"], ["00002", "00003"]])

    assert set(classes.values()) == {"word00001-00001"}
    assert len(classes) == 3


def test_sign_classes_ignores_entries_that_were_not_crawled():
    classes = sign_classes(entries("00002"), [["00002", "09999"]])  # 09999 has no entry with a video

    assert classes == {}  # one crawled clip is not a class


def write_crawl(raw_dir, rows, groups):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "entries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (raw_dir / "groups.jsonl").write_text("\n".join(json.dumps(group) for group in groups))


def test_read_videos(tmp_path):
    write_crawl(
        tmp_path,
        [
            {"id": "00001", "word": "a", "video": "/movies/00/a-00001-tecken.mp4", "same_form": True},
            {"id": "00002", "word": "b", "video": "/movies/00/b-00002-tecken.mp4", "same_form": True},
            {"id": "00003", "word": "c", "video": "/movies/00/c-00003-tecken.mp4", "same_form": False},
            {"id": "00004", "word": None, "video": None, "same_form": None},
        ],
        [{"id": "00001", "members": ["00001", "00002"]}],
    )

    videos = read_videos(tmp_path)

    assert videos["clip_id"].to_list() == ["00001", "00002"]  # only the class, not the single clips
    assert videos["sign"].to_list() == ["a-00001", "a-00001"]
    assert videos["signer"].to_list() == [None, None]  # the lexicon publishes no signer ids
    assert videos["path"].to_list() == [f"{tmp_path}/movies/00/a-00001-tecken.mp4", f"{tmp_path}/movies/00/b-00002-tecken.mp4"]  # fmt: skip


def test_shown_clips():
    clips = pl.DataFrame({"clip_id": ["00003", "00001", "00002", "00007", "00005"], "sign": ["a", "a", "a", "b", "b"]})

    assert sorted(shown_clips(clips)) == ["00001", "00005"]  # the lowest lexicon id of each class
