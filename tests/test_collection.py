import polars as pl
import pytest

from isolated_sign_validation.collection import NO_EVENT, NO_EVENT_PROMPTS, Recordings, session_prompts
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
