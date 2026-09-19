"""Extract the recordings' landmarks in a browser and compare them with the Python extraction.

A future application could extract landmarks on the user's device and send only those, not the
video. This runs web/extract.html (MediaPipe's HolisticLandmarker for the web, same version, model
and settings as extraction.py) in headless Google Chrome over every video of a landmark store,
frame by frame at the store's frame rate, and writes the result as a second store with the same
clips. It then prints, per landmark group, how often the two extractions disagree on whether the
group was detected, and how far apart the landmarks are where both detected it, in shoulder widths.

Run from the repo root: uv run scripts/compare_browser_extraction.py
Then prepare and evaluate the browser store like the Python one, to compare what the model sees:
    uv run scripts/prepare_recordings.py --store data/processed/recordings_browser
    uv run scripts/evaluate_recordings.py --run <run> --prepared data/prepared/recordings_browser-<id> ...
"""

import argparse
import base64
import functools
import http.server
import threading
from pathlib import Path

import numpy as np
import polars as pl
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from isolated_sign_validation.collection import RAW_DIR
from isolated_sign_validation.datasets import recordings
from isolated_sign_validation.extraction import MODEL_PATH, download_model
from isolated_sign_validation.landmarks import LANDMARK_SLICES, N_LANDMARKS, LandmarkStore, write_store

SHOULDERS = [LANDMARK_SLICES["pose"].start + 11, LANDMARK_SLICES["pose"].start + 12]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    """Serve `directory` on localhost at a free port, in a background thread."""
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def extract_in_browser(store: LandmarkStore, paths: dict[str, str]) -> list[tuple[dict, np.ndarray]]:
    """(metadata, landmarks) of every clip of `store`, extracted from its video in headless Chrome."""
    server = serve(Path.cwd())
    root = f"http://127.0.0.1:{server.server_port}"
    clips = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")  # Google Chrome decodes H.264; Chromium doesn't
        page = browser.new_page()
        page.on("console", lambda message: print("browser:", message.text) if message.type == "error" else None)
        page.goto(f"{root}/web/extract.html?model=/{MODEL_PATH}")
        page.wait_for_function("window.extractLandmarks !== undefined", timeout=120_000)
        for i, metadata in enumerate(tqdm(store.clips.to_dicts(), desc="browser", unit="clip")):
            n_frames, fps = len(store[i]), metadata["fps"]
            result = page.evaluate("([url, fps, n]) => extractLandmarks(url, fps, n)", [f"{root}/{paths[metadata['clip_id']]}", fps, n_frames])
            media_times = np.array(result["mediaTimes"])
            if not np.allclose(media_times, np.arange(n_frames) / fps, atol=0.25 / fps):
                raise RuntimeError(f"{metadata['clip_id']}: the browser showed other frames than 0..{n_frames - 1}: {media_times[:5]}...")
            if (result["width"], result["height"]) != (metadata["width"], metadata["height"]):
                raise RuntimeError(f"{metadata['clip_id']}: frame size {result['width']}x{result['height']} in the browser")
            landmarks = np.frombuffer(base64.b64decode(result["landmarks"]), dtype=np.float32).reshape(n_frames, N_LANDMARKS, 3)
            clips.append((metadata, landmarks))
        browser.close()
    server.shutdown()
    return clips


def compare(python: LandmarkStore, browser: LandmarkStore) -> pl.DataFrame:
    """Per landmark group: frames where only one extraction detected it, and the distance between
    the two extractions' landmarks where both did, in shoulder widths of the Python extraction."""
    rows = []
    for i, metadata in enumerate(python.clips.to_dicts()):
        size = np.array([metadata["width"], metadata["height"]])
        a, b = python[i][..., :2] * size, browser[i][..., :2] * size  # pixels
        shoulder_width = np.nanmedian(np.linalg.norm(a[:, SHOULDERS[0]] - a[:, SHOULDERS[1]], axis=1))
        for group, s in LANDMARK_SLICES.items():
            found_a, found_b = np.isfinite(a[:, s, 0]).all(axis=1), np.isfinite(b[:, s, 0]).all(axis=1)
            both = found_a & found_b
            distance = np.linalg.norm(a[both, s] - b[both, s], axis=-1) / shoulder_width
            rows.append({"group": group, "frames": len(a), "either": int((found_a | found_b).sum()), "disagree": int((found_a != found_b).sum()), "distances": distance.ravel().tolist()})  # fmt: skip
    return (
        pl.DataFrame(rows)
        .group_by("group", maintain_order=True)
        .agg(pl.col("frames", "either", "disagree").sum(), pl.col("distances").list.explode(keep_nulls=False, empty_as_null=False))
        .with_columns(
            median=pl.col("distances").list.median(),
            p95=pl.col("distances").list.eval(pl.element().quantile(0.95)).list.first(),
            max=pl.col("distances").list.max(),
        )
        .drop("distances")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=recordings.STORE_DIR, help="the Python extraction")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="the videos the store was extracted from")
    parser.add_argument("--out", type=Path, default=Path("data/processed/recordings_browser"))
    args = parser.parse_args()

    download_model()
    python = LandmarkStore(args.store)
    paths = dict(recordings.read_videos(args.raw_dir).select("clip_id", "path").iter_rows())
    print(f"{len(python)} clips, {python.clips['n_frames'].sum()} frames -> {args.out}")
    write_store(args.out, extract_in_browser(python, paths))
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(compare(python, LandmarkStore(args.out)))


if __name__ == "__main__":
    main()
