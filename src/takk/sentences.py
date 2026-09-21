"""Writing the Swedish sentence that a practice turn is signed in.

TAKK signs the key words of a spoken sentence, so practising several signs at once means a sentence
that contains them. The sentence depends on what this learner has due, which is only known when the
request arrives, so it is written per request by an LLM rather than taken from a prepared list (see
the decisions in ROADMAP-takk.md).

The words of the turn are offered to the model rather than imposed on it: the first one, which is
what the pass has scheduled next, must appear, and it picks from the rest whichever belong with it in
one idea. Deciding the combination here produced sentences that mentioned each word in turn instead
of being about anything. Every word it does use must appear exactly as written: the signs are found in the recording by timing those words in the audio
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

MOST_WORDS = 3  # key words in one sentence; every extra sign is another boundary the split can misplace

SYSTEM = """You write one short Swedish sentence for someone practising TAKK (tecken som alternativ och kompletterande kommunikation).

The user gives you a list of words they are practising. The first one must appear in your sentence. The others are offered, not required: use the ones that belong with it in a single everyday idea, and leave out the ones that would only fit by being listed. A sentence about one thing ("Har du en röd tröja?") teaches more than one that mentions two words in turn ("Nallen har en röd napp och ett öga."). Use at most {most} of the words in all, including the first.

Every word you use from the list must appear exactly once and spelled exactly as given: no inflection, no compounding, no capitalisation change except a sentence-initial capital. Swedish makes this easy — use the infinitive after a modal verb ("vill äta", "kan hjälpa"), and an indefinite noun after an article ("en hund", "mer mjölk").

Write the rest of the sentence freely in natural, everyday Swedish. Keep it short, concrete and easy to say aloud, the kind of sentence said to a small child at home. It has to be spoken while signing, so twelve words is already long.""".format(most=MOST_WORDS)


class Written(BaseModel):
    """What the model answers with. The order of the words is read off the sentence, not reported."""

    sentence: str


def key_words(sentence: str, offered: list[str], most: int = MOST_WORDS, require_first: bool = True) -> list[str] | None:  # fmt: skip
    """The words of `offered` that `sentence` uses, in the order they occur in it. The set is read off
    the sentence rather than reported, since a whole-word search finds it exactly: "mer" is not
    satisfied by "mera", and "blå" is not found inside "blåbär".

    None when the sentence cannot be used: the first word is the one the turn is for and has to be
    there, no word may be used twice, since a repeated word leaves it unclear which occurrence is
    signed, and at most `most` of them can be signed in one recording. A part of a story sets
    `require_first`: its words are offered with none of them scheduled, so any of them will do
    (`story.py`)."""
    at = {}
    for word in offered:
        found = [match.start() for match in re.finditer(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE)]
        if len(found) > 1 or (not found and require_first and word == offered[0]):
            return None
        if found:
            at[word] = found[0]
    if not at or len(at) > most:
        return None
    return sorted(at, key=at.__getitem__)


def write_sentence(client: anthropic.Anthropic, offered: list[str], model: str = MODEL, tries: int = 2) -> tuple[str, list[str]] | None:  # fmt: skip
    """A Swedish sentence using the first of `offered` and whichever of the rest belong with it, and
    those words in the order they occur in it. None when the model does not manage it within `tries`,
    which leaves the caller to practise the words one at a time instead.

    The model chooses the combination because it is the one that knows which words make a sentence:
    asked to put "röd" and "blå" in one, it reaches for two objects to colour, while offered a list
    it can pick the pair that belongs together and leave the rest for another turn. Only the first
    word is required, so the turn still practises what the pass has scheduled next."""
    asked = f"Jag övar på orden: {', '.join(offered)}. Skriv en mening med {offered[0]} i."
    messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": asked}]
    for _ in range(tries):
        response = client.messages.parse(model=model, max_tokens=1000, system=SYSTEM, messages=messages, output_format=Written)  # fmt: skip
        sentence = response.parsed_output.sentence
        used = key_words(sentence, offered)
        if used is not None:
            return sentence, used
        # The model is told what was wrong rather than asked again blindly: the failure is almost
        # always a word inflected, left out or used twice, which it can fix in the sentence it wrote.
        messages += [
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": f"Meningen måste innehålla {offered[0]}, och varje ord ur listan som den använder ska stå exakt en gång och exakt som det är skrivet, högst {MOST_WORDS} av dem. Skriv om den."},  # fmt: skip
        ]
    return None
