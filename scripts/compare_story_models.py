"""Which model and effort should write the story a learner signs their way through (`takk/story.py`).

One request per word list and configuration, measured on what actually matters: whether the story
comes back usable on the first ask (`story_parts` accepts it, so no retry doubles the wait), how long
the ask takes, and how many tokens it spends. The retry loop of `write_story` is deliberately not run
here - a configuration that needs it has already lost the latency it was being compared on.

Costs real money: one full sweep is 4 configurations x 10 word lists = 40 requests.

Run from the repo root, with ANTHROPIC_API_KEY set:
    uv run scripts/compare_story_models.py
"""

import argparse
import time

import anthropic
import polars as pl

from takk.story import SYSTEM, Told, story_parts

# What a learner's boxes hold: everyday TAKK words, the weakest ones a story would be asked to lean
# on. Ten lists of the size `POST /api/story` is called with, drawn from the starter vocabulary.
WORD_LISTS = [
    ["mamma", "mjölk", "mer", "sova", "bil"],
    ["pappa", "äta", "bröd", "vatten", "ute"],
    ["hund", "katt", "springa", "glad", "stor"],
    ["bok", "läsa", "säng", "natt", "kram"],
    ["blå", "röd", "boll", "kasta", "leka"],
    ["skor", "jacka", "ut", "kallt", "vante"],
    ["äpple", "banan", "smörgås", "hungrig", "tallrik"],
    ["bada", "tvätta", "handduk", "varm", "tvål"],
    ["farmor", "hälsa", "komma", "fika", "kaka"],
    ["ledsen", "arg", "trött", "hjälpa", "vila"],
]


def ask(client: anthropic.Anthropic, model: str, effort: str, words: list[str], parts: int) -> dict:
    """One ask, as `write_story` makes its first one, with what it cost and whether it is usable."""
    asked = f"Jag övar på orden: {', '.join(words)}. Skriv en berättelse i {parts} delar."
    began = time.monotonic()
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": asked}],
        output_config={"effort": effort},
        output_format=Told,
    )
    seconds = time.monotonic() - began
    told = response.parsed_output
    return {
        "model": model,
        "effort": effort,
        "words": ", ".join(words),
        "usable": told is not None and story_parts(told.parts, words, parts) is not None,
        "seconds": seconds,
        "stop_reason": response.stop_reason,
        "output_tokens": response.usage.output_tokens,
        "story": " ".join(told.parts) if told else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-5", "claude-opus-5"])
    parser.add_argument("--efforts", nargs="+", default=["low", "medium"], help="output_config.effort")
    parser.add_argument("--parts", type=int, default=6, help="parts per ask; 6 is the page's own default")
    parser.add_argument("--lists", type=int, default=len(WORD_LISTS), help="how many of the word lists to use")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    rows = []
    for model in args.models:
        for effort in args.efforts:
            for words in WORD_LISTS[: args.lists]:
                row = ask(client, model, effort, words, args.parts)
                rows.append(row)
                print(f"{model} {effort:6} {'ok  ' if row['usable'] else 'fail'} {row['seconds']:5.1f} s  {row['story'][:60]}")  # fmt: skip

    trials = pl.DataFrame(rows)
    trials.write_ndjson("outputs/story_models.jsonl")
    print()
    print(
        trials.group_by("model", "effort")
        .agg(
            pl.col("usable").mean().alias("first_try"),
            pl.col("seconds").median().alias("median_s"),
            pl.col("seconds").max().alias("worst_s"),
            pl.col("output_tokens").mean().alias("output_tokens"),
        )
        .sort("model", "effort")
    )
    print("\nEvery ask in outputs/story_models.jsonl")


if __name__ == "__main__":
    main()
