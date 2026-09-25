"""Writing the Swedish story a learner signs their way through, a few sentences at a time.

A story is how everything already learned is repeated (step 12 of ROADMAP-takk.md): one text is
written around many of the words in the learner's Leitner boxes, and it is practised in order from
beginning to end. The words are the ones
in the learner's boxes, drawn towards the low ones, so the weak words are the ones the story leans
on; nothing new is taught here, since words are learned on their own ("Ord", step 9).

The story comes back in chunks ("avsnitt"), one recording each. A chunk carries at most `MOST_WORDS`
signs: every sign is another boundary the split has to place, and the whole chunk has to be said in
one breath-length recording.

The words are inflected as the sentence needs them, which is how TAKK is actually used, so the model
reports what it wrote rather than the text being searched for base forms (user, 2026-09-25): each
sign of a part says the form as it stands in the text (`said`, "bilen") and the offered word it signs
(`word`, "bil"). `signed_words` checks what can be checked — the form is really in the part, in the
order given, and the word is one that was offered — and trusts the pairing itself, since no cheap
rule relates a Swedish surface form to its base ("åt" shares nothing with "äta"). The form is what is
spoken, so it is what `speech.py` times in the audio and what the page marks; the word is what names
the sign and moves the box. A word said twice in a part is signed twice and reported twice: forced
alignment is positional, so two occurrences are two targets and two spans.
"""

import re

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-sonnet-5"

# How deeply the model thinks before it writes. Low: measured over 10 word lists, thinking longer
# bought nothing a story needs (see the findings in ROADMAP-takk.md).
EFFORT = "low"

MOST_WORDS = 3  # signs in one part; every extra sign is another boundary the split can misplace

SYSTEM = """You write a short Swedish story for someone practising TAKK (tecken som alternativ och kompletterande kommunikation).

The user gives you the words they are practising and how many parts the story should have. Aim for that many parts. Each part is one or two short sentences, and the parts follow each other as one story with a beginning and an end.

Each part signs between one and {most} of the user's words, counting a word signed twice as two. Spread the words over the story so that every word is used somewhere, and use a word again in a later part when the story has a reason to.

Write in natural, everyday Swedish and inflect the user's words as the sentence needs them: "bilen", "sover" and "åt" are all good Swedish and all fine here. For every one of the user's words that a part uses, report the sign it is: `said` is the form exactly as it stands in that part's text, and `word` is the word from the user's list that it signs. Report the signs in the order they are spoken, and report a word twice when the part says it twice.

`said` must be a form of `word` and nothing else. A compound that merely contains the word is a different sign, so "blåbär" is never the word "blå".

Write the rest freely: concrete, easy to say aloud, the kind of story told to a small child. Each part is spoken while its words are signed, so keep a part to what can be said in one breath.""".format(most=MOST_WORDS)


class Signed(BaseModel):
    """One sign of a part: the form as the part says it, and the offered word that form signs."""

    said: str
    word: str


class Part(BaseModel):
    """A part of the story: the Swedish text to say aloud, and its signs in the order they are
    spoken."""

    text: str
    signs: list[Signed]


class Told(BaseModel):
    """What the model answers with: the story's parts, in order."""

    parts: list[Part]


def signed_words(part: Part, offered: list[str], most: int = MOST_WORDS) -> list[Signed] | None:
    """The signs of `part`, or None when it cannot be practised: no signs at all or more than `most`
    of them, a word that was never offered, or a form the part does not say where it claims to.

    Each form is looked for as a whole word from the end of the previous one, so the signs are also
    checked to be in the order they are spoken - which is the order the audio is cut in - and a form
    said twice needs two signs to be marked and scored twice."""
    if not 1 <= len(part.signs) <= most:
        return None
    cursor = 0
    for sign in part.signs:
        if sign.word not in offered:
            return None
        found = re.compile(rf"\b{re.escape(sign.said)}\b", re.IGNORECASE).search(part.text, cursor)
        if found is None:
            return None
        cursor = found.end()
    return part.signs


def story_parts(parts: list[Part], offered: list[str]) -> list[Part] | None:
    """The story's parts when every one of them can be practised, or None when one cannot. How many
    parts there are is not checked: the number asked for is what the story aims at, and a story one
    part short is a shorter pass rather than a story to throw away."""
    if not parts:
        return None
    if any(signed_words(part, offered) is None for part in parts):
        return None
    return parts


def write_story(client: anthropic.Anthropic, words: list[str], parts: int, model: str = MODEL, tries: int = 2) -> list[Part] | None:  # fmt: skip
    """A Swedish story of about `parts` parts over `words`, each part with the signs it is practised
    by. None when the model does not manage it within `tries`, which leaves the caller with nothing
    to tell."""
    asked = f"Jag övar på orden: {', '.join(words)}. Skriv en berättelse i {parts} delar."
    messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": asked}]
    for _ in range(tries):
        # One rule for every way an ask can come to nothing: it is a try that failed, never an error
        # raised at the learner. The API can refuse to answer at all (rate limited, overloaded, timed
        # out) and an answer cut off by `max_tokens` can carry JSON the parse refuses; the response
        # can also carry no text block at all, since the model thinks before it writes and a turn cut
        # off in the thinking ends there, which is what `parsed_output` being None means.
        try:
            response = client.messages.parse(model=model, max_tokens=16000, system=SYSTEM, messages=messages, output_config={"effort": EFFORT}, output_format=Told)  # fmt: skip
        except (anthropic.APIError, ValidationError) as failed:
            print(f"the story writer could not answer: {failed}")
            continue
        if response.parsed_output is None:
            continue
        told = response.parsed_output.parts
        story = story_parts(told, words)
        if story is not None:
            return story
        # The model is told what was wrong rather than asked again blindly,
        # since the failure is almost always a sign reported for a form the part does not say.
        messages += [
            {"role": "assistant", "content": "\n".join(part.text for part in told)},
            {"role": "user", "content": f"Varje del ska teckna minst ett och högst {MOST_WORDS} av orden ur listan. För varje tecken ska `said` stå ordagrant i delens text, i den ordning de sägs, och `word` vara ordet ur listan som tecknas. Skriv om den."},  # fmt: skip
        ]
    return None
