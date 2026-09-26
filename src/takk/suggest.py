"""The chips on Sök: a few named sets of words fitted to who the learner is (step 23 of
ROADMAP-takk.md).

The learner says, on Om dig, whether they have signed before and, in their own words, who they sign
with and where. The model answers with a handful of chips, each a label and the Swedish words under
it, and a chip is pressed to fill the grid the search fills. A chip carries its own words rather than
being a query for `vocabulary.search`, since a label the model writes ("Påklädning") is not the name
of one of the lexicon's categories, and the categories are subject areas rather than what a learner
starts from (see the findings).

Each word is looked up exactly among the words the lexicon has a sign for, the most counted first,
and a word it has no sign for is simply dropped, as is a chip left empty: nothing here is worth
asking again for, so there is no validation to fail and no second try.
"""

import anthropic
from pydantic import BaseModel, ValidationError

from takk.story import EFFORT, MODEL
from takk.vocabulary import Index, spoken_word

CHIPS = 6  # what the model is asked for; a chip that resolves to no sign is dropped

SYSTEM = """You help someone start learning TAKK (tecken som alternativ och kompletterande kommunikation), which signs the key words of spoken Swedish with signs from Svenskt teckenspråkslexikon.

The user says whether they have signed before, who they sign with and where, and which words they have already chosen. Suggest {chips} groups of words for them to learn next, each with a short Swedish label of one or two words ("Maten", "Känslor", "På förskolan") and about ten words.

Choose the words that person will actually need in their everyday life, the most useful first. When they have never signed, the first group is "Första orden": the few words everyone signs from the first day, such as ja, nej, mer, klar, hjälp, äta, dricka. Leave out the words they have already chosen.

Write every word as a dictionary headword: lowercase, uninflected, and one word where one word will do ("äta", not "äter"; "hund", not "hundar")."""


class Chip(BaseModel):
    label: str
    words: list[str]


class Suggested(BaseModel):
    chips: list[Chip]


LEVELS = {"ny": "Jag har aldrig tecknat.", "lite": "Jag har tecknat lite.", "van": "Jag tecknar redan."}


def resolve(chips: list[Chip], vocabulary: Index) -> list[dict]:
    """Each chip with its words as the search offers them (sign, entry, word), dropping the words
    the lexicon has no sign for, a sign already in the chip, and the chips left with nothing."""
    lookup: dict[str, dict] = {}
    for word in vocabulary.words:  # most counted first, so a homograph resolves to its common sign
        lookup.setdefault((word.get("word") or spoken_word(word["sign"])).lower(), word)
    resolved = []
    for chip in chips:
        found: dict[str, dict] = {}
        for word in (w.strip().lower() for w in chip.words):
            if word in lookup:
                found.setdefault(lookup[word]["sign"], lookup[word])
        if found:
            resolved.append({"label": chip.label, "words": list(found.values())})
    return resolved


def suggest(client: anthropic.Anthropic, vocabulary: Index, level: str, about: str, known: list[str], model: str = MODEL) -> list[dict]:  # fmt: skip
    """The chips for a learner at `level` ("ny", "lite" or "van") who describes themselves as
    `about`, and who has already chosen `known`. Empty when the model does not answer."""
    asked = "\n".join(
        [LEVELS.get(level, ""), f"Om mig: {about.strip() or 'inget särskilt'}.", f"Ord jag redan valt: {', '.join(known) or 'inga'}."]  # fmt: skip
    )
    try:
        response = client.messages.parse(model=model, max_tokens=16000, system=SYSTEM.format(chips=CHIPS), messages=[{"role": "user", "content": asked}], output_config={"effort": EFFORT}, output_format=Suggested)  # fmt: skip
    except (anthropic.APIError, ValidationError) as failed:
        print(f"the chips could not be suggested: {failed}")
        return []
    return resolve(response.parsed_output.chips, vocabulary) if response.parsed_output else []
