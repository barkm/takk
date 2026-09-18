"""Crawl Svenskt teckenspråkslexikon into data/raw/sts-lexikon.

The lexicon has no bulk download, so the entries are crawled from the site in three phases, each of
which skips what an earlier run already has, so an interrupted run resumes when run again:

1. `/ord/<id>` for every id up to `--max_id`, appended to `entries.jsonl` (the ids are sparse, and
   an id that is not a published entry is recorded with null fields so it is not fetched again).
2. `/ord/<id>/kan-aven-betyda` for every entry that shares its sign form with others, appended to
   `groups.jsonl`. These groups become the sign classes (see `datasets/sts_lexikon.py`).
3. The `-tecken.mp4` of every entry in a class of at least two clips, into `movies/`, mirroring the
   paths on the site. `--all_videos` downloads the whole vocabulary instead (~21,700 clips, ~15 GB).

The lexicon is CC BY-NC-SA 4.0 and its robots.txt allows crawling; keep `--workers` modest anyway.
The site is updated continuously, so note the crawl date when reporting results.

Run from the repo root:
    uv run scripts/download_sts_lexikon.py
    uv run scripts/download_sts_lexikon.py --max_id 2000  # a small sample of the lexicon
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from isolated_sign_validation.datasets.sts_lexikon import (
    BASE_URL,
    ENTRIES_FILE,
    GROUPS_FILE,
    RAW_DIR,
    parse_entry,
    parse_group,
    sign_classes,
)

USER_AGENT = "isolated-sign-validation research crawler"
MAX_ID = 26_999  # the highest published id was below 26,300 in 2026-09; ids above it simply 404


def fetch(url: str, retries: int = 3) -> bytes | None:
    """The body at `url`, or None if it doesn't exist. Retries a few times on a transient failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}")


def fetch_page(url: str) -> str | None:
    """The page at `url` as text, or None if it doesn't exist."""
    body = fetch(url)
    return None if body is None else body.decode("utf-8", "replace")


def crawl[T](items: Sequence[T], work: Callable[[T], dict], workers: int, out: Path, desc: str) -> None:
    """Run `work` over `items` in threads and append each returned row to the JSON lines file `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool, out.open("a") as file:
        for row in tqdm(pool.map(work, items), total=len(items), desc=desc, unit="page"):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


def crawled_ids(path: Path) -> set[str]:
    """The lexicon ids already in the JSON lines file `path`, which may not exist yet."""
    return set(pl.read_ndjson(path)["id"]) if path.exists() else set()


def crawl_entries(raw_dir: Path, max_id: int, workers: int) -> None:
    """Fetch the entry pages up to `max_id` that are not in entries.jsonl yet."""
    out = raw_dir / ENTRIES_FILE
    done = crawled_ids(out)
    todo = [f"{entry_id:05d}" for entry_id in range(1, max_id + 1) if f"{entry_id:05d}" not in done]
    print(f"{len(done)} entry pages crawled, {len(todo)} to go")
    crawl(todo, lambda i: {"id": i, **parse_entry(fetch_page(f"{BASE_URL}/ord/{i}"))}, workers, out, "entries")


def crawl_groups(raw_dir: Path, workers: int) -> None:
    """Fetch the same-form group page of every entry that has one and is not in groups.jsonl yet."""
    out = raw_dir / GROUPS_FILE
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE)
    with_form = entries.filter(pl.col("video").is_not_null() & pl.col("same_form"))["id"]
    todo = sorted(set(with_form) - crawled_ids(out))
    print(f"{with_form.len()} entries share their form with others, {len(todo)} group pages to go")

    def members(entry_id: str) -> dict:
        page = fetch_page(f"{BASE_URL}/ord/{entry_id}/kan-aven-betyda")
        return {"id": entry_id, "members": parse_group(page) if page else []}

    crawl(todo, members, workers, out, "groups")


def video_paths(raw_dir: Path, all_videos: bool) -> Iterator[str]:
    """The site paths of the videos to download: the sign classes' clips, or the whole vocabulary."""
    entries = pl.read_ndjson(raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    if all_videos:
        return iter(entries["video"])
    classes = sign_classes(entries, pl.read_ndjson(raw_dir / GROUPS_FILE)["members"].to_list())
    return iter(entries.filter(pl.col("id").is_in(list(classes)))["video"])


def download_video(video: str, raw_dir: Path) -> None:
    """Download the sign video at the site path `video` into `raw_dir`, unless it is already there."""
    path = raw_dir / video.lstrip("/")
    if path.exists() and path.stat().st_size:
        return
    body = fetch(BASE_URL + video)
    if body is None:
        raise FileNotFoundError(f"{BASE_URL}{video} is gone; crawl the entries again")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")  # a killed run must not leave a truncated video behind
    partial.write_bytes(body)
    partial.rename(path)


def download_videos(raw_dir: Path, all_videos: bool, workers: int) -> None:
    """Download the videos that are not in `raw_dir` yet."""
    videos = [video for video in video_paths(raw_dir, all_videos) if not (raw_dir / video.lstrip("/")).exists()]
    print(f"{len(videos)} videos to download")
    with ThreadPoolExecutor(workers) as pool:
        for _ in tqdm(pool.map(lambda video: download_video(video, raw_dir), videos), total=len(videos), unit="clip"):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--max_id", type=int, default=MAX_ID, help="the highest lexicon id to try")
    parser.add_argument("--workers", type=int, default=8, help="parallel requests")
    parser.add_argument("--all_videos", action="store_true", help="download every entry's video, not only the classes'")
    args = parser.parse_args()

    crawl_entries(args.raw_dir, args.max_id, args.workers)
    crawl_groups(args.raw_dir, args.workers)
    download_videos(args.raw_dir, args.all_videos, args.workers)

    entries = pl.read_ndjson(args.raw_dir / ENTRIES_FILE).filter(pl.col("video").is_not_null())
    classes = sign_classes(entries, pl.read_ndjson(args.raw_dir / GROUPS_FILE)["members"].to_list())
    downloaded = sum(path.stat().st_size for path in (args.raw_dir / "movies").rglob("*.mp4"))
    print(
        f"{entries.height} entries with a sign video, {len(set(classes.values()))} signs with at least two clips "
        f"({len(classes)} clips), {downloaded / 2**30:.1f} GB downloaded"
    )


if __name__ == "__main__":
    main()
