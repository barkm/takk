import polars as pl

from isolated_sign_validation.collection import SCHEMA
from isolated_sign_validation.datasets.recordings import read_videos


def test_read_videos_takes_only_the_kept_takes(tmp_path):
    pl.DataFrame(
        [
            {"clip_id": "s_HELLO_1", "signer": "A", "sign": "HELLO", "kept": False},
            {"clip_id": "s_HELLO_2", "signer": "A", "sign": "HELLO", "kept": True},
            {"clip_id": "s_no_event_1", "signer": "B", "sign": "no_event", "kept": True},
        ],
        schema=SCHEMA,
    ).write_csv(tmp_path / "clips.csv")

    videos = read_videos(tmp_path)

    assert videos["clip_id"].to_list() == ["s_HELLO_2", "s_no_event_1"]
    assert videos["sign"].to_list() == ["HELLO", "no_event"]
    assert videos["signer"].to_list() == ["recordings:A", "recordings:B"]
    assert videos["path"].to_list() == [f"{tmp_path}/videos/s_HELLO_2.mp4", f"{tmp_path}/videos/s_no_event_1.mp4"]
