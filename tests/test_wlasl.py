import json
import subprocess

import numpy as np
import polars as pl

from isolated_sign_validation.datasets.wlasl import map_signs, read_videos, sign_labels, trim_clip, twin_pairs
from isolated_sign_validation.splits import sign_split


def test_read_videos_skips_instances_without_a_video(tmp_path):
    videos_dir = tmp_path / "data" / "data_0"
    videos_dir.mkdir(parents=True)
    (videos_dir / "00584.mp4").touch()
    (videos_dir / "63452.mp4").touch()
    glosses = [
        {"gloss": "accent", "instances": [{"video_id": "00584", "signer_id": 4, "url": "a"}, {"video_id": "11111", "signer_id": 7, "url": "b"}]},
        {"gloss": "wine", "instances": [{"video_id": "63452", "signer_id": 11, "url": "c"}]},
    ]
    (tmp_path / "WLASL_v0.3.json").write_text(json.dumps(glosses))

    videos = read_videos(tmp_path)

    assert videos["clip_id"].to_list() == ["00584", "63452"]  # the instance without a video is skipped
    assert videos["sign"].to_list() == ["accent", "wine"]
    assert videos["signer"].to_list() == ["wlasl:4", "wlasl:11"]
    assert videos["url"].to_list() == ["a", "c"]
    assert videos["path"][0] == str(videos_dir / "00584.mp4")


def test_map_signs():
    # WINE is a val sign, the others are train signs (by the hash split, checked below)
    assert sign_split("WINE") != "train" and sign_split("FATHER") == sign_split("CHICKEN") == "train"
    asl_citizen = ["FATHER", "DRINK1", "DRINK2", "WINE", "POLICEMAN1"]

    mapping = map_signs(["father", "drink", "wine", "chicken", "policeman"], asl_citizen)

    assert mapping["father"] == "FATHER"  # a unique match joins the ASL Citizen class
    assert mapping["policeman"] == "POLICEMAN1"  # matching ignores the sense number
    assert mapping["drink"] is None  # several variants, the label doesn't say which
    assert mapping["wine"] is None  # a held-out sign
    assert mapping["chicken"] == "CHICKEN"  # a new sign the hash split puts in train


def test_sign_labels_keep_held_out_signs():
    labels = sign_labels(["wine", "drink", "chicken"], ["WINE", "DRINK1", "DRINK2"])

    assert labels == {"wine": "WINE", "drink": "DRINK1|DRINK2", "chicken": "CHICKEN"}


def test_twin_pairs():
    clips = pl.DataFrame(
        {
            "label": ["HABIT", "TRADITION", "TRADITION", "RAIN", "RAIN", "LAST", "FINAL", "PAST"],
            "url": ["u1", "u1", "u2", "u3", "u3", "u4", "u4", "u5"],
        }
    )

    # a shared video makes a pair; a video of one label doesn't
    assert twin_pairs(clips) == {("HABIT", "TRADITION"), ("FINAL", "LAST")}


def test_map_signs_holds_out_twins_of_held_out_signs():
    assert sign_split("WINE") != "train" and sign_split("FATHER") == sign_split("CHICKEN") == "train"

    assert sign_split("SHORT1") == "train" and sign_split("SHORT2") != "train"
    twins = {("CHICKEN", "WINE"), ("FATHER", "SHORT1|SHORT2")}

    mapping = map_signs(["father", "wine", "chicken", "short"], ["FATHER", "WINE", "SHORT1", "SHORT2"], twins)

    assert mapping["chicken"] is None  # may show the held-out WINE under another word
    assert mapping["father"] is None  # may show the held-out variant SHORT2
    assert mapping["short"] is None


def _gray_video(path, values, size=16):
    """A video whose frame i is a flat gray of values[i]."""
    frames = np.repeat(np.asarray(values, np.uint8), size * size).tobytes()
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size}x{size}", "-r", "25"]
    subprocess.run([*command, "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)], input=frames, check=True)


def _gray_values(path, size=16):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    frames = np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, np.uint8)
    return frames.reshape(-1, size * size).mean(axis=1).round().astype(int).tolist()


def test_trim_clip_keeps_the_inclusive_1_based_frame_range(tmp_path):
    source = tmp_path / "source.mkv"
    _gray_video(source, [8 * i for i in range(30)])

    trim_clip(source, tmp_path / "a.mp4", frame_start=11, frame_end=20)
    trim_clip(source, tmp_path / "b.mp4", frame_start=1, frame_end=-1)

    expected = _gray_values(source)[10:20]  # frames 11..20
    assert np.abs(np.subtract(_gray_values(tmp_path / "a.mp4"), expected)).max() <= 2
    assert len(_gray_values(tmp_path / "b.mp4")) == 30  # -1: to the end
    assert not list(tmp_path.glob("*.part*"))
