"""Browse the ASL Citizen twin candidates side by side in a browser and label each pair.

Reads outputs/results/asl_citizen_twin_candidates.csv (written by scripts/find_twins.py) and serves a
page showing one pair at a time: a row of clips per sign, the same signer in a column where someone
signed both. Keys: 1 same form, 2 minimal pair, 3 different, 4 unsure, left/right previous/next pair,
n next unlabeled pair, space restart the clips. Labels are saved to outputs/asl_citizen_twins/labels.csv
as they are given, and a restart picks up where they left off.

Run from the repo root: uv run scripts/inspect_twins.py
Then open http://localhost:8000 (VS Code Remote-SSH forwards the port).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import asl_citizen

CANDIDATES = Path("outputs/results/asl_citizen_twin_candidates.csv")
LABELS = Path("outputs/asl_citizen_twins/labels.csv")
VIDEOS = asl_citizen.RAW_DIR / "videos"
COLUMNS = 5  # clips shown per sign

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Twin candidates</title>
<style>
  body { font: 15px system-ui, sans-serif; margin: 12px; background: #111; color: #ddd; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .info { color: #aaa; margin-bottom: 8px; }
  .row { display: flex; gap: 6px; align-items: flex-start; margin-bottom: 6px; }
  .row b { width: 130px; flex: none; font-size: 18px; }
  figure { margin: 0; width: 240px; }
  video { width: 240px; background: #000; }
  figcaption { font-size: 12px; color: #999; }
  figcaption.shared { color: #fc6; }
  .verdict { font-size: 18px; margin: 8px 0; }
  .same { color: #6d6; } .minimal { color: #6af; } .different { color: #e66; } .unsure { color: #fc6; }
  button { font-size: 15px; margin-right: 4px; }
</style>
<h1 id="title"></h1>
<div class="info" id="info"></div>
<div id="rows"></div>
<div class="verdict" id="verdict"></div>
<div>
  <button data-v="same">1 same form</button><button data-v="minimal">2 minimal pair</button>
  <button data-v="different">3 different</button><button data-v="unsure">4 unsure</button>
  <button id="prev">&larr;</button><button id="next">&rarr;</button><button id="unlabeled">n next unlabeled</button>
  speed <select id="speed"><option>0.5</option><option>0.75</option><option selected>1</option></select>
</div>
<script>
let pairs = [], i = 0;
const $ = id => document.getElementById(id);

function show() {
  const p = pairs[i];
  location.hash = i + 1;
  const labeled = pairs.filter(q => q.verdict).length;
  $("title").textContent = `${i + 1}. ${p.sign_a} / ${p.sign_b}`;
  $("info").textContent = `splits ${p.split_a} / ${p.split_b} · signals ${p.signals}` +
    ` (${[p.wlasl && "WLASL", p.asl_lex && "ASL-LEX"].filter(Boolean).join(", ") || "similarity only"})` +
    ` · gap model ${p.model_gap.toFixed(3)}, hand ${p.hand_gap.toFixed(3)} · ASL-LEX differs in: ${p.phonology_differs}` +
    ` · ${labeled} of ${pairs.length} labeled`;
  $("rows").innerHTML = [["sign_a", "clips_a"], ["sign_b", "clips_b"]].map(([s, c]) =>
    `<div class="row"><b>${p[s]}</b>` + p[c].map(clip =>
      `<figure><video src="/video/${clip.file}" autoplay muted loop playsinline></video>` +
      `<figcaption class="${clip.shared ? "shared" : ""}">signer ${clip.signer}</figcaption></figure>`
    ).join("") + "</div>").join("");
  const rate = +$("speed").value;
  document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  $("verdict").textContent = p.verdict || "";
  $("verdict").className = "verdict " + (p.verdict || "");
}

async function label(verdict) {
  const p = pairs[i];
  p.verdict = verdict;
  await fetch("/label", {method: "POST", body: JSON.stringify({sign_a: p.sign_a, sign_b: p.sign_b, verdict})});
  go(i + 1);
}

function go(j) { i = Math.max(0, Math.min(pairs.length - 1, j)); show(); }
function nextUnlabeled() { const j = pairs.findIndex((q, k) => k > i && !q.verdict); if (j >= 0) go(j); }

document.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => label(b.dataset.v));
$("prev").onclick = () => go(i - 1);
$("next").onclick = () => go(i + 1);
$("unlabeled").onclick = nextUnlabeled;
$("speed").onchange = () => document.querySelectorAll("video").forEach(v => v.playbackRate = +$("speed").value);
document.onkeydown = e => {
  const verdicts = {"1": "same", "2": "minimal", "3": "different", "4": "unsure"};
  if (verdicts[e.key]) label(verdicts[e.key]);
  else if (e.key === "ArrowLeft") go(i - 1);
  else if (e.key === "ArrowRight") go(i + 1);
  else if (e.key === "n") nextUnlabeled();
  else if (e.key === " ") { e.preventDefault(); document.querySelectorAll("video").forEach(v => { v.currentTime = 0; v.play(); }); }
};

fetch("/pairs.json").then(r => r.json()).then(data => {
  pairs = data;
  const start = parseInt(location.hash.slice(1)) - 1;
  if (start >= 0) go(start); else { i = -1; nextUnlabeled(); if (i < 0) go(0); }
});
</script>
"""


def clips_by_column(videos: dict[str, dict[str, str]], a: str, b: str, n: int = COLUMNS) -> tuple[list[dict], list[dict]]:
    """n clips of each sign, one per signer; signers who signed both first, in the same columns.
    `videos` maps each sign to one video file per signer."""
    both = sorted(videos[a].keys() & videos[b].keys())[:n]
    result = []
    for sign in (a, b):
        signers = both + sorted(videos[sign].keys() - set(both))
        result.append([{"file": videos[sign][s], "signer": s, "shared": s in both} for s in signers[:n]])
    return result[0], result[1]


def read_labels() -> dict[tuple[str, str], str]:
    if not LABELS.exists():
        return {}
    return {(a, b): v for a, b, v in pl.read_csv(LABELS).iter_rows()}


def write_labels(labels: dict[tuple[str, str], str]) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    rows = [(a, b, v) for (a, b), v in labels.items()]
    pl.DataFrame(rows, schema=["sign_a", "sign_b", "verdict"], orient="row").write_csv(LABELS)


def make_handler(pairs_json: bytes, labels: dict[tuple[str, str], str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/pairs.json":
                pairs = json.loads(pairs_json)
                for pair in pairs:
                    pair["verdict"] = labels.get((pair["sign_a"], pair["sign_b"]))
                self.send(200, json.dumps(pairs).encode(), "application/json")
            elif self.path.startswith("/video/") and (path := VIDEOS / Path(self.path).name).exists():
                data = path.read_bytes()
                start, end = 0, len(data) - 1
                if header := self.headers.get("Range"):  # browsers seek and loop videos with range requests
                    first, _, last = header.removeprefix("bytes=").partition("-")
                    start, end = int(first or 0), int(last) if last else end
                    headers = {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{len(data)}"}
                    self.send(206, data[start : end + 1], "video/mp4", headers)
                else:
                    self.send(200, data, "video/mp4", {"Accept-Ranges": "bytes"})
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            labels[body["sign_a"], body["sign_b"]] = body["verdict"]
            write_labels(labels)
            self.send(204, b"", "text/plain")

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    candidates = pl.read_csv(CANDIDATES)
    videos = {}
    for sign, signer, file in asl_citizen.read_videos(asl_citizen.RAW_DIR).select("Gloss", "Participant ID", "Video file").iter_rows():
        videos.setdefault(sign, {}).setdefault(signer, file)
    pairs = []
    for row in candidates.iter_rows(named=True):
        row["clips_a"], row["clips_b"] = clips_by_column(videos, row["sign_a"], row["sign_b"])
        pairs.append(row)
    labels = read_labels()
    print(f"{len(pairs)} pairs, {len(labels)} labeled; open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(json.dumps(pairs).encode(), labels)).serve_forever()


if __name__ == "__main__":
    main()
