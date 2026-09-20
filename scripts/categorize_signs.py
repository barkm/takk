"""Ask Claude for the everyday Swedish words of each category, and keep the ones the lexicon signs.

The practice app needs a beginner's word sets to build lessons from, and the lexicon crawl has no
categories of its own (see step 6 of ROADMAP-takk.md). The words are asked for rather than the
lexicon labelled: a TAKK learner starts from a few hundred everyday signs, not from the lexicon's
16,026, and a word the model invents that the lexicon has no sign for is simply dropped. One request
per category, so a full run costs cents.

The result is committed as `src/takk/categories.json` and read from there at run time, because the
same question has the same answer every time; the sentences of step 9 are what an LLM writes per
request. A run replaces the categories it was asked for and leaves the others as they are.

Needs an API key in ANTHROPIC_API_KEY (or a profile from `ant auth login`). Run from the repo root:
    uv run scripts/categorize_signs.py
    uv run scripts/categorize_signs.py --categories "mat och dryck" "djur"  # only these, again
"""

import argparse
import json
from pathlib import Path

import anthropic
import polars as pl

from takk.categories import PATH, sign_by_word

MODEL = "claude-opus-5"

# What a TAKK learner starts from: everyday things to name, said and signed with a small child.
CATEGORIES = {
    "mat och dryck": "det som äts och dricks till vardags",
    "familj och personer": "familjemedlemmar och personer i barnets närhet",
    "känslor och behov": "hur man mår och vad man vill",
    "kläder": "kläder och skor man tar på sig",
    "kropp och hälsa": "kroppsdelar, att må bra och att må dåligt",
    "djur": "djur man möter hemma, i böcker och utomhus",
    "leka och leksaker": "lek, leksaker och spel",
    "hemma och vardagsrutiner": "saker och rutiner i hemmet, som att sova, äta och tvätta",
    "förskola och skola": "förskolans dag och det som finns där",
    "natur och utomhus": "väder, natur och att vara ute",
    "tid och mängd": "tidsord och mängdord i vardagen",
    "transport": "att åka och färdas",
    "vanliga verb": "de vanligaste handlingarna i vardagen",
    "artighet och samspel": "hälsningar, artighet och att turas om",
}

SYSTEM = """Du hjälper till att bygga en app där föräldrar och personal övar TAKK, tecken som \
alternativ och kompletterande kommunikation, på svenska. I TAKK tecknas nyckelorden i talad svenska \
med tecken ur det svenska teckenspråkslexikonet, så orden ska vara ord man faktiskt säger i \
vardagen tillsammans med ett litet barn.

Svara med ett ord per rad och ingenting annat: ingen numrering, inga bindestreck, inga \
förklaringar. Skriv varje ord i grundform och med små bokstäver, ett enda ord per rad."""


def ask(client: anthropic.Anthropic, model: str, category: str, hint: str, n: int) -> list[str]:
    """The `n` most useful Swedish words of `category`, as the model ranks them."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,  # a list of at most a few dozen single words
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Kategori: {category} ({hint}). Ge de {n} vanligaste orden i den kategorin, de som en nybörjare har mest nytta av först."}],  # fmt: skip
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return [line.strip().lower() for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glossary", type=Path, default=Path("data/prepared/sts_lexikon-234f4575"), help="prepared evaluation set whose signs the words must exist in")  # fmt: skip
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES), metavar="CATEGORY", help="the categories to ask for again; all of them by default")  # fmt: skip
    parser.add_argument("--words", type=int, default=25, help="words asked for per category, before the ones the lexicon has no sign for are dropped")  # fmt: skip
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, default=PATH)
    args = parser.parse_args()

    signs = sign_by_word(pl.read_parquet(args.glossary / "clips.parquet")["sign"].unique().to_list())
    sets = json.loads(args.out.read_text()) if args.out.exists() else {}
    client = anthropic.Anthropic()
    for category in args.categories:
        words = ask(client, args.model, category, CATEGORIES[category], args.words)
        sets[category] = [signs[w] for w in dict.fromkeys(words) if w in signs]
        missing = [w for w in words if w not in signs]
        print(f"{category}: {len(sets[category])} tecken" + (f", utan tecken i lexikonet: {', '.join(missing)}" if missing else ""))  # fmt: skip
    args.out.write_text(json.dumps({c: sets[c] for c in CATEGORIES if c in sets}, ensure_ascii=False, indent=2) + "\n")
    print(f"skrev {sum(len(s) for s in sets.values())} tecken i {len(sets)} kategorier till {args.out}")


if __name__ == "__main__":
    main()
