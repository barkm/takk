"""Writing the Swedish sentence that a practice turn is signed in.

TAKK signs the key words of a spoken sentence, so practising several signs at once means a sentence
that contains them. The sentence depends on what this learner has due, which is only known when the
request arrives, so it is written per request by an LLM rather than taken from a prepared list (see
the decisions in ROADMAP-takk.md).

The words to practise are given, so the model only writes around them, and it must use each of them
exactly as written: the signs are found in the recording by timing those words in the audio
(`speech.py`), and a form that was not spoken as written scores below `MIN_WORD_SCORE` and is
refused. Forbidding inflection keeps the whole contract checkable — every word appears as a whole
token, once — rather than trusting the model to report which form it chose. Swedish gives that up
cheaply: the dictionary form of a verb is the infinitive, so "Jag vill äta" and "Kan du hjälpa mig"
are already base forms, and a noun sits in its base form behind an article ("Jag ser en hund").

Only the key words have to be signs at all. The rest of the sentence is spoken and not signed, so
the model is free to write any Swedish around them and nothing about the learner's vocabulary has to
be sent.
"""

import re

import anthropic
from pydantic import BaseModel

MODEL = "claude-sonnet-5"

SYSTEM = """You write one short Swedish sentence for someone practising TAKK (tecken som alternativ och kompletterande kommunikation).

The sentence must contain every word the user lists, each exactly once and spelled exactly as given: no inflection, no compounding, no capitalisation change except a sentence-initial capital. Swedish makes this easy — use the infinitive after a modal verb ("vill äta", "kan hjälpa"), and an indefinite noun after an article ("en hund", "mer mjölk").

Write the rest of the sentence freely in natural, everyday Swedish. Keep it short, concrete and easy to say aloud, the kind of sentence said to a small child at home. It has to be spoken while signing, so twelve words is already long."""


class Written(BaseModel):
    """What the model answers with. The order of the words is read off the sentence, not reported."""

    sentence: str


def key_order(sentence: str, words: list[str]) -> list[str] | None:
    """`words` in the order they occur in `sentence`, or None when the sentence does not use each of
    them exactly once as a whole word. The order matters because it is the order the signs are made
    in, and the match is whole-word so that "mer" is not satisfied by "mera"."""
    at = {}
    for word in words:
        found = [match.start() for match in re.finditer(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE)]
        if len(found) != 1:
            return None
        at[word] = found[0]
    return sorted(words, key=at.__getitem__)


def write_sentence(client: anthropic.Anthropic, words: list[str], model: str = MODEL, tries: int = 2) -> tuple[str, list[str]] | None:  # fmt: skip
    """A Swedish sentence using each of `words` once, and the words in the order they occur in it.
    None when the model does not manage it within `tries`, which leaves the caller to practise the
    words one at a time instead."""
    asked = "Skriv en mening som innehåller orden: " + ", ".join(words)
    messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": asked}]
    for _ in range(tries):
        response = client.messages.parse(model=model, max_tokens=1000, system=SYSTEM, messages=messages, output_format=Written)  # fmt: skip
        sentence = response.parsed_output.sentence
        order = key_order(sentence, words)
        if order is not None:
            return sentence, order
        # The model is told what was wrong rather than asked again blindly: the failure is almost
        # always one word inflected or left out, which it can fix in the sentence it just wrote.
        messages += [
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": f"Meningen använder inte vart och ett av orden {', '.join(words)} exakt en gång och exakt som det är skrivet. Skriv om den."},  # fmt: skip
        ]
    return None
