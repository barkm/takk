import json

import polars as pl
import pytest
from fastapi import HTTPException

from isolated_sign_validation.collection import NO_EVENT, NO_EVENT_PROMPTS, Recordings, create_app, lexicon_prompts, session_prompts
from isolated_sign_validation.preparation import PrepConfig
from isolated_sign_validation.splits import sign_split


def write_asl_citizen(raw_dir, glosses):
    """A stand-in for the ASL Citizen splits: every gloss by three signers, two videos each."""
    rows = [
        {"Participant ID": f"P{signer}", "Video file": f"{gloss}_{signer}_{take}.mp4", "Gloss": gloss}
        for gloss in glosses
        for signer in range(3)
        for take in range(2)
    ]
    (raw_dir / "splits").mkdir(parents=True)
    schema = pl.Schema({column: pl.String for column in rows[0]})
    for split in ("train", "val", "test"):
        videos = pl.DataFrame(rows if split == "train" else rows[:0], schema=schema)
        videos.write_csv(raw_dir / "splits" / f"{split}.csv")


def test_session_prompts(tmp_path):
    candidates = [f"SIGN{i}" for i in range(500)]
    held_out = [gloss for gloss in candidates if sign_split(gloss) == "test"]
    write_asl_citizen(tmp_path, candidates)

    prompts = session_prompts(tmp_path, n_signs=4, takes=3, n_references=2, seed=0)

    signs = [prompt["sign"] for prompt in prompts if prompt["sign"] != NO_EVENT]
    assert len(signs) == 12 and set(signs) <= set(held_out)  # only signs the model has not been trained on
    assert all(count == 3 for count in pl.Series(signs).value_counts()["count"])
    assert prompts == session_prompts(tmp_path, n_signs=4, takes=3, n_references=2, seed=0)  # same signs for every signer

    for prompt in prompts:
        if prompt["sign"] == NO_EVENT:
            assert prompt["references"] == [] and prompt["instruction"]
        else:
            signers = {reference.split("_")[1] for reference in prompt["references"]}
            assert len(prompt["references"]) == 2 and len(signers) == 2  # references by different signers

    rest = [i for i, prompt in enumerate(prompts) if prompt["sign"] == NO_EVENT]
    assert len(rest) == len(NO_EVENT_PROMPTS) and min(rest) > 0  # spread through the session, never first


def row(**overrides):
    defaults = {
        "clip_id": "s_HELLO_1", "session": "s", "signer": "A", "handedness": "right", "sign": "HELLO",
        "take": 1, "recorded_at": "2026-09-17T12:00:00", "usable": True, "note": "Looks good.",
        "hand_share": 0.9, "kept": False, "confident": False,
    }  # fmt: skip
    return defaults | overrides


def test_recordings_counts_takes_within_a_session(tmp_path):
    recordings = Recordings(tmp_path)
    assert recordings.take("s", "HELLO") == 1

    recordings.add(row())
    assert recordings.take("s", "HELLO") == 2  # a discarded take still counts
    assert recordings.take("s", "APPLE") == 1 and recordings.take("other", "HELLO") == 1


def test_recordings_keep_survives_a_restart(tmp_path):
    recordings = Recordings(tmp_path)
    recordings.add(row(clip_id="s_HELLO_1"))
    recordings.add(row(clip_id="s_HELLO_2", take=2))
    recordings.keep("s_HELLO_2", confident=True)

    clips = Recordings(tmp_path).clips
    assert clips["kept"].to_list() == [False, True]
    assert clips["confident"].to_list() == [False, True]
    assert clips["take"].to_list() == [1, 2] and clips["hand_share"].to_list() == [0.9, 0.9]


def test_recordings_keep_rejects_an_unknown_clip(tmp_path):
    with pytest.raises(KeyError):
        Recordings(tmp_path).keep("nope", confident=False)


def write_lexicon(raw_dir):
    """A stand-in for the crawled lexicon: classes of 3 and 2 clips, and an entry of its own."""
    entries = [
        {"id": entry_id, "word": word, "video": f"/movies/00/{word}-{entry_id}-tecken.mp4", "same_form": True}
        for entry_id, word in [("00003", "a"), ("00001", "b"), ("00002", "c"), ("00007", "d"), ("00005", "e"), ("00009", "f")]
    ]
    groups = [{"id": "00001", "members": ["00001", "00002", "00003"]}, {"id": "00005", "members": ["00005", "00007"]}]
    (raw_dir / "entries.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries))
    (raw_dir / "groups.jsonl").write_text("\n".join(json.dumps(group) for group in groups))


def test_lexicon_prompts(tmp_path):
    write_lexicon(tmp_path)

    prompts = lexicon_prompts(tmp_path, n_signs=2, takes=3, seed=0)

    signs = [prompt for prompt in prompts if prompt["sign"] != NO_EVENT]
    assert len(signs) == 6 and len(prompts) == 6 + len(NO_EVENT_PROMPTS)
    # one clip shown per class, its lowest id; the class's other clips are what it is scored against
    assert {prompt["sign"]: prompt["references"][0] for prompt in signs} == {
        "sts:b-00001": "movies/00/b-00001-tecken.mp4",
        "sts:e-00005": "movies/00/e-00005-tecken.mp4",
    }


def test_reference_route_serves_only_shown_clips(tmp_path):
    prompts = [{"sign": "sts:b-00001", "instruction": None, "references": ["movies/00/b-00001-tecken.mp4"]}]
    app = create_app(prompts, Recordings(tmp_path / "recordings"), tmp_path, PrepConfig())
    route = next(route for route in app.routes if getattr(route, "path", "").startswith("/api/reference/"))

    assert route.path_regex.match("/api/reference/movies/00/b-00001-tecken.mp4")  # a nested path is one name
    assert str(route.endpoint("movies/00/b-00001-tecken.mp4").path) == str(tmp_path / "movies/00/b-00001-tecken.mp4")
    with pytest.raises(HTTPException):
        route.endpoint("../outside.mp4")
