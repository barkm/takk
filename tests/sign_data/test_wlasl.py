import json

import polars as pl

from sign_data.datasets.wlasl import read_videos, sign_labels, twin_pairs


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
