"""The glossary the app serves, as one directory that carries no training data.

A bundle is what `scripts/build_serving.py` writes from a published verifier and a prepared
glossary, and the only thing the API reads at startup: the verifier (`verifier/`) and the glossary's
signs as its references (`references.parquet`), both in the verifier's own formats, and beside them
what the app itself knows of the glossary - the addresses its clips are watched at and the words a
learner can search for. About 40 MB, against the 641 MB of prepared frames the glossary is built from.

Reading one needs the verifier API (`isolated_sign_verification.verifier`) and nothing else of the
training package, which is what lets the deployed image leave the data packages out.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from isolated_sign_verification.verifier import References, Verifier
from takk.vocabulary import Index, make_index

VERIFIER_DIR = "verifier"
REFERENCES_FILE = "references.parquet"
CLIPS_FILE = "clips.parquet"
VOCABULARY_FILE = "vocabulary.json"
META_FILE = "meta.json"


@dataclass(frozen=True)
class Bundle:
    """Everything the API serves: the verifier and the glossary's signs as its references, each sign's
    clip addresses, the words that lead to them and the lexicon's description of each entry's form."""

    verifier: Verifier
    references: References
    clips: dict[str, list[str]]
    vocabulary: Index
    forms: dict[str, str]
    meta: dict


def write(
    path: Path,
    verifier: Verifier,
    references: References,
    clips: dict[str, list[str]],
    vocabulary: Index,
    forms: dict[str, str],
    source: dict[str, str],
) -> None:
    """Write a bundle. `source` names the prepared glossary the references came from and the
    verifier, which is kept in `meta.json` beside the verifier's own record of its run, so a served
    model can always be traced back to what made it."""
    path.mkdir(parents=True, exist_ok=True)
    verifier.save(path / VERIFIER_DIR)
    references.save(path / REFERENCES_FILE)
    pl.DataFrame(
        {"sign": references.signs, "clips": [clips.get(sign, []) for sign in references.signs]},
        schema={"sign": pl.String, "clips": pl.List(pl.String)},
    ).write_parquet(path / CLIPS_FILE)
    (path / VOCABULARY_FILE).write_text(
        json.dumps({"words": vocabulary.words, "themes": vocabulary.themes, "forms": forms}, ensure_ascii=False)
    )
    meta = source | {"created": date.today().isoformat(), "n_signs": len(references.signs)}
    (path / META_FILE).write_text(json.dumps(meta, indent=2))


def load(path: Path, device: str = "cpu", threshold: float | None = None) -> Bundle:
    """Read a bundle. `threshold` overrides the verifier's."""
    table = pl.read_parquet(path / CLIPS_FILE)
    vocabulary = json.loads((path / VOCABULARY_FILE).read_text())
    return Bundle(
        verifier=Verifier.load(path / VERIFIER_DIR, device, threshold),
        references=References.load(path / REFERENCES_FILE),
        clips=dict(zip(table["sign"], table["clips"].to_list())),
        vocabulary=make_index(vocabulary["words"], vocabulary["themes"]),
        forms=vocabulary["forms"],
        meta=json.loads((path / META_FILE).read_text()),
    )
