import json

import numpy as np
import polars as pl
import pytest
from fastapi import HTTPException

from isolated_sign_validation.collection import NO_EVENT, NO_EVENT_PROMPTS, SCHEMA, Recordings, check_clip, create_app, overlay, read_clips, session_prompts
from isolated_sign_validation.extraction import VideoInfo
from isolated_sign_validation.landmarks import N_LANDMARKS
from isolated_sign_validation.preparation import PrepConfig


def glossary(n_signs, signers):
    """A stand-in for the test split of a prepared evaluation set: every sign by `signers`, two clips each."""
    rows = [
        {"clip_id": f"SIGN{sign}_{signer}_{take}", "sign": f"SIGN{sign}", "signer": signer}
        for sign in range(n_signs)
        for signer in signers
        for take in range(2)
    ]
    return pl.DataFrame(rows, schema={"clip_id": pl.String, "sign": pl.String, "signer": pl.String})


def test_session_prompts():
    clips = glossary(50, ["P1", "P2", "P3"])

    prompts = session_prompts(clips, n_signs=4, takes=3, n_references=2, seed=0)

    signs = [prompt["sign"] for prompt in prompts if prompt["sign"] != NO_EVENT]
    assert len(signs) == 12 and set(signs) <= set(clips["sign"])  # only the glossary's signs
    assert all(count == 3 for count in pl.Series(signs).value_counts()["count"])
    assert prompts == session_prompts(clips, n_signs=4, takes=3, n_references=2, seed=0)  # same signs for every signer

    for prompt in prompts:
        if prompt["sign"] == NO_EVENT:
            assert prompt["references"] == [] and prompt["instruction"]
        else:
            shown = clips.filter(pl.col("clip_id").is_in(prompt["references"]))
            assert len(prompt["references"]) == 2 and shown["signer"].n_unique() == 2  # references by different signers
            assert (shown["sign"] == prompt["sign"]).all()  # glossary clips of the prompted sign

    rest = [i for i, prompt in enumerate(prompts) if prompt["sign"] == NO_EVENT]
    assert len(rest) == len(NO_EVENT_PROMPTS) and min(rest) > 0  # spread through the session, never first


def test_session_prompts_without_signer_ids():
    prompts = session_prompts(glossary(5, [None]), n_signs=2, takes=1, n_references=3, seed=0)

    # a glossary without signer ids counts as one signer, so one clip is shown
    assert all(len(prompt["references"]) == 1 for prompt in prompts if prompt["sign"] != NO_EVENT)


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


def test_read_clips_fills_columns_added_later(tmp_path):
    Recordings(tmp_path).add(row())
    older = pl.read_csv(tmp_path / "clips.csv").drop("references")  # a file from before the column existed
    older.write_csv(tmp_path / "clips.csv")

    clips = read_clips(tmp_path / "clips.csv")

    assert clips.columns == list(SCHEMA) and clips["references"].to_list() == [None]
    assert Recordings(tmp_path).take("s", "HELLO") == 2  # and the app still opens it


def test_reference_route_serves_only_shown_clips(tmp_path):
    prompts = [{"sign": "sts:b-00001", "instruction": None, "references": ["00001"]}]
    app = create_app(prompts, Recordings(tmp_path / "recordings"), {"00001": str(tmp_path / "b-00001.mp4")}, PrepConfig())
    route = next(route for route in app.routes if getattr(route, "path", "").startswith("/api/reference/"))

    assert str(route.endpoint("00001").path) == str(tmp_path / "b-00001.mp4")
    with pytest.raises(HTTPException):
        route.endpoint("00002")


def test_check_clip_accepts_not_signing_without_hands():
    no_hands = np.full((60, N_LANDMARKS, 3), np.nan, dtype=np.float32)  # someone sitting still, hands out of view
    info = VideoInfo(30.0, 640, 480)

    assert check_clip(no_hands, info, PrepConfig()) == (False, "No hands were detected. Are your hands inside the frame while signing?", 0.0)
    assert check_clip(no_hands, info, PrepConfig(), signing=False) == (True, "Recorded.", 0.0)


def test_overlay_draws_the_skeletons_with_undetected_points_as_null():
    landmarks = np.full((2, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    landmarks[0, 468] = (0.25, 0.5, 0.0)  # the left hand's wrist, in the first frame only
    landmarks[0, 469] = (0.75, 0.125, 0.0)

    result = overlay(landmarks)

    wrist, thumb = result["edges"]["left_hand"][0]  # MediaPipe's first hand connection is wrist to thumb
    assert result["frames"][0][2 * wrist : 2 * wrist + 2] == [0.25, 0.5]
    assert result["frames"][0][2 * thumb : 2 * thumb + 2] == [0.75, 0.125]
    assert result["frames"][1][2 * wrist] is None
    assert json.dumps(result, allow_nan=False)  # no NaN, which the JSON response would reject
