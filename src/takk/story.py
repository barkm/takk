"""Writing the Swedish story a learner signs their way through, a few sentences at a time.

A story is how everything already learned is repeated (step 12 of ROADMAP-takk.md): one text is
written around many of the words in the learner's Leitner boxes, and it is practised in order from
beginning to end. The words are the ones
in the learner's boxes, drawn towards the low ones, so the weak words are the ones the story leans
on; nothing new is taught here, since words are learned on their own ("Ord", step 9).

The story comes back in chunks ("avsnitt"), one recording each. A chunk carries at most `MOST_WORDS`
signs: every sign is another boundary the split has to place, and the whole chunk has to be said in
one breath-length recording. `key_words` checks the contract: a key word appears exactly once in its
chunk and exactly as written, since the signs are found by timing those words in the audio
(`speech.py`), and a form that was not spoken as written scores below `MIN_WORD_SCORE` and is
refused. Forbidding inflection keeps it checkable — every word appears as a whole token, once —
rather than trusting the model to report which form it chose. Swedish gives that up cheaply: the
dictionary form of a verb is the infinitive, so "vill äta" is already the base form, and a noun sits
in its base form behind an article ("en hund").
"""

import re

import anthropic
from pydantic import BaseModel

MODEL = "claude-sonnet-5"

MOST_WORDS = 3  # key words in one part; every extra sign is another boundary the split can misplace

SYSTEM = """You write a short Swedish story for someone practising TAKK (tecken som alternativ och kompletterande kommunikation).

The user gives you the words they are practising and how many parts the story should have. Write the story in exactly that many parts. Each part is one or two short sentences, and the parts follow each other as one story with a beginning and an end.

Each part must use between one and {most} of the user's words. Spread the words over the story so that every word is used somewhere, and use a word again in a later part when the story has a reason to.

Every word you use from the list must appear exactly once in its own part and spelled exactly as given: no inflection, no compounding, no capitalisation change except a sentence-initial capital. Swedish makes this easy — use the infinitive after a modal verb ("vill äta", "kan hjälpa"), and an indefinite noun after an article ("en hund", "mer mjölk").

Write the rest freely in natural, everyday Swedish: concrete, easy to say aloud, the kind of story told to a small child. Each part is spoken while its words are signed, so keep a part to what can be said in one breath.""".format(most=MOST_WORDS)


class Told(BaseModel):
    """What the model answers with: the story's parts, in order, as plain Swedish sentences."""

    parts: list[str]


def key_words(part: str, offered: list[str], most: int = MOST_WORDS) -> list[str] | None:
    """The words of `offered` that `part` uses, in the order they occur in it. The set is read off the
    text rather than reported, since a whole-word search finds it exactly: "mer" is not satisfied by
    "mera", and "blå" is not found inside "blåbär".

    None when the part cannot be practised: no word may be used twice, since a repeated word leaves
    it unclear which occurrence is signed, and at most `most` of them can be signed in one
    recording."""
    at = {}
    for word in offered:
        found = [match.start() for match in re.finditer(rf"\b{re.escape(word)}\b", part, re.IGNORECASE)]
        if len(found) > 1:
            return None
        if found:
            at[word] = found[0]
    if not at or len(at) > most:
        return None
    return sorted(at, key=at.__getitem__)


Chunk = tuple[str, list[str]]  # a part of the story and the offered words it uses, in spoken order


def story_parts(parts: list[str], offered: list[str], asked: int) -> list[Chunk] | None:
    """The story's parts with the words each one uses, or None when it cannot be practised: the wrong
    number of parts, a part that signs nothing, or a part whose words `key_words` refuses (inflected,
    used twice, or more than `MOST_WORDS` of them)."""
    if len(parts) != asked:
        return None
    chunks = []
    for part in parts:
        used = key_words(part, offered)
        if not used:
            return None
        chunks.append((part, used))
    return chunks


def write_story(client: anthropic.Anthropic, words: list[str], parts: int, model: str = MODEL, tries: int = 2) -> list[Chunk] | None:  # fmt: skip
    """A Swedish story of `parts` parts over `words`, each part with the words it uses in the order
    they are spoken. None when the model does not manage it within `tries`, which leaves the caller
    to practise the words in a daily pass instead."""
    asked = f"Jag övar på orden: {', '.join(words)}. Skriv en berättelse i {parts} delar."
    messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": asked}]
    for _ in range(tries):
        response = client.messages.parse(model=model, max_tokens=16000, system=SYSTEM, messages=messages, output_format=Told)  # fmt: skip
        # No parsed answer means the response carries no text block at all - the model thinks before
        # it writes, and a turn cut off by `max_tokens` (or declined) ends with the thinking alone.
        # That is a try that failed like any other, not an error to raise at the learner.
        if response.parsed_output is None:
            continue
        told = response.parsed_output.parts
        story = story_parts(told, words, parts)
        if story is not None:
            return story
        # The model is told what was wrong rather than asked again blindly,
        # since the failure is almost always a word inflected, left out of a part or used twice.
        messages += [
            {"role": "assistant", "content": "\n".join(told)},
            {"role": "user", "content": f"Berättelsen ska ha exakt {parts} delar, och varje del ska använda minst ett och högst {MOST_WORDS} av orden ur listan, vart och ett exakt en gång och exakt som det är skrivet. Skriv om den."},  # fmt: skip
        ]
    return None
