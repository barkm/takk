import json

from isolated_sign_validation.datasets.mm_wlauslan import read_videos


def test_read_videos(tmp_path):
    labels = tmp_path / "Annotation" / "Labels & Split"
    labels.mkdir(parents=True)
    (labels / "Train.json").write_text(json.dumps({"35776": "WHALE", "10920": "TURN ON (START)"}))
    (labels / "Test_STU.json").write_text(json.dumps({"75205": "WHALE"}))

    videos = read_videos(tmp_path, ["Train", "Test_STU"])

    assert videos["subset"].to_list() == ["Train", "Train", "Test_STU"]
    assert videos["clip_id"].to_list() == ["35776", "10920", "75205"]
    assert videos["sign"].to_list() == ["WHALE", "TURN ON (START)", "WHALE"]
    assert videos["path"][2] == str(tmp_path / "Test-STU" / "Kinect_F" / "rgb" / "75205_kf_rgb.mp4")
