import json

from isolated_sign_validation.datasets.wlasl import map_signs, read_videos
from isolated_sign_validation.splits import sign_split


def test_read_videos_skips_instances_without_a_video(tmp_path):
    videos_dir = tmp_path / "data" / "data_0"
    videos_dir.mkdir(parents=True)
    (videos_dir / "00584.mp4").touch()
    (videos_dir / "63452.mp4").touch()
    glosses = [
        {"gloss": "accent", "instances": [{"video_id": "00584", "signer_id": 4}, {"video_id": "11111", "signer_id": 7}]},
        {"gloss": "wine", "instances": [{"video_id": "63452", "signer_id": 11}]},
    ]
    (tmp_path / "WLASL_v0.3.json").write_text(json.dumps(glosses))

    videos = read_videos(tmp_path)

    assert videos["clip_id"].to_list() == ["00584", "63452"]  # the instance without a video is skipped
    assert videos["sign"].to_list() == ["accent", "wine"]
    assert videos["signer"].to_list() == ["wlasl:4", "wlasl:11"]
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
