import json

from isolated_sign_validation.datasets.mm_wlauslan import read_videos, sign_words


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


def test_sign_words(tmp_path):
    folder = tmp_path / "WWW_CV_ISLR_Challenge" / "Dictionary_Mapping"
    folder.mkdir(parents=True)
    entry = {"Group_Name": "TURN ON (START)", "Keywords": "light,illuminate,turn on (start)", "State": "AustraliaWide-traditional"}
    (folder / "Dictionary.json").write_text(json.dumps({"TURN ON (START)": entry}))
    assert sign_words(tmp_path) == {"TURN ON (START)": ["TURN ON (START)", "light", "illuminate", "turn on (start)"]}
