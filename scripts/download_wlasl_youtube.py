"""Download WLASL's YouTube instances, which the Kaggle mirror lacks, into data/raw/wlasl/youtube.

All 5,135 YouTube instances (3,250 videos) are missing from the mirror, and about three in four of
the videos are still public (see ROADMAP.md). Each video is downloaded once with yt-dlp, every
instance of it is cut out at its frame range into `youtube/<video id>.mp4` (where
`datasets/wlasl.py` finds it like the mirror's clips), and the downloaded video is deleted. A video
that can't be had (private, removed) is recorded in `youtube/unavailable.tsv` with yt-dlp's error
and skipped from then on; an instance whose frame range runs past the end of its video is skipped
too. A run stops when YouTube asks to confirm that it isn't a bot. Clips that exist are skipped, so
an interrupted run resumes when run again.

Run from the repo root:
    uv run scripts/download_wlasl_youtube.py
    uv run scripts/download_wlasl_youtube.py --max_videos 20  # a sample
"""

import argparse
import json
import subprocess
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from isolated_sign_validation.datasets.wlasl import METADATA_FILE, RAW_DIR, trim_clip

OUT_DIR = RAW_DIR / "youtube"
UNAVAILABLE_FILE = OUT_DIR / "unavailable.tsv"
FORMAT = "bv*[height<=1080]/b[height<=1080]/b"  # the mirror's clips go up to 1080p


class BotCheck(Exception):
    """YouTube wants a signed-in user; every further request would fail the same way."""


def youtube_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.path.strip("/") if parsed.netloc.endswith("youtu.be") else urllib.parse.parse_qs(parsed.query)["v"][0]


def missing_instances(raw_dir: Path) -> dict[str, list[dict]]:
    """The YouTube instances without a clip anywhere below `raw_dir`, by YouTube id."""
    have = {path.stem for path in raw_dir.rglob("*.mp4")}
    by_video = {}
    for gloss in json.loads((raw_dir / METADATA_FILE).read_text()):
        for instance in gloss["instances"]:
            if "youtu" in instance["url"] and instance["video_id"] not in have:
                by_video.setdefault(youtube_id(instance["url"]), []).append(instance)
    return by_video


def download(video: str, out_dir: Path) -> Path:
    """Download the YouTube video `video` into `out_dir`; raises RuntimeError with yt-dlp's error."""
    command = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--js-runtimes", "node", "-f", FORMAT]
    command += ["-o", str(out_dir / "%(id)s.%(ext)s"), "--", video]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        error = result.stderr.strip().splitlines()[-1]
        raise (BotCheck if "confirm you" in error else RuntimeError)(error)
    return next(path for path in out_dir.glob(f"{video}.*") if not path.name.endswith(".part"))


def frame_count(path: Path) -> int:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets"]
    command += ["-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)]
    return int(subprocess.run(command, capture_output=True, text=True, check=True).stdout)


def fetch_video(video: str, instances: list[dict], out_dir: Path, stop: threading.Event) -> tuple[int, int, str | None]:
    """Download `video` and cut out its `instances`: (clips written, instances out of range, error)."""
    if stop.is_set():
        return 0, 0, None
    try:
        source = download(video, out_dir / "source")
    except BotCheck:
        stop.set()
        raise
    except RuntimeError as error:
        return 0, 0, str(error)
    frames = frame_count(source)
    fits = [i for i in instances if max(i["frame_start"], i["frame_end"]) <= frames]
    for instance in fits:
        trim_clip(source, out_dir / f"{instance['video_id']}.mp4", instance["frame_start"], instance["frame_end"])
    source.unlink()
    return len(fits), len(instances) - len(fits), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="videos downloaded in parallel")
    parser.add_argument("--max_videos", type=int, help="only this many videos, for a sample")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unavailable = {line.split("\t")[0] for line in UNAVAILABLE_FILE.read_text().splitlines()} if UNAVAILABLE_FILE.exists() else set()  # fmt: skip
    todo = {video: instances for video, instances in missing_instances(RAW_DIR).items() if video not in unavailable}
    videos = sorted(todo)[: args.max_videos]
    print(f"{len(unavailable)} videos known unavailable, {len(todo)} videos to try, {len(videos)} this run")

    stop, clips, out_of_range, failed = threading.Event(), 0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool, UNAVAILABLE_FILE.open("a") as file:
        results = pool.map(lambda video: (video, fetch_video(video, todo[video], OUT_DIR, stop)), videos)
        for video, (written, skipped, error) in tqdm(results, total=len(videos), unit="video"):
            clips, out_of_range = clips + written, out_of_range + skipped
            if error:
                failed += 1
                file.write(f"{video}\t{error}\n")
                file.flush()
    print(f"{clips} clips written, {failed} videos unavailable, {out_of_range} instances past the end of their video")


if __name__ == "__main__":
    main()
