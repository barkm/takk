from types import SimpleNamespace

import anthropic

from takk.suggest import Chip, Suggested, resolve, suggest
from takk.vocabulary import Index

# most counted first, as `search_index` orders them: "blå" is a word of the sign named for "öga"
WORDS = [
    {"sign": "sts:äta-1", "id": "1"},
    {"sign": "sts:öga-2", "id": "3", "word": "blå"},
    {"sign": "sts:öga-2", "id": "2"},
    {"sign": "sts:hund-4", "id": "4"},
    {"sign": "sts:hund-9", "id": "9"},  # a rarer homograph of "hund"
]
VOCABULARY = Index(WORDS, {}, {})


def test_a_chip_keeps_the_words_the_lexicon_has_a_sign_for():
    chips = [Chip(label="Maten", words=["Äta", "gurkmajonnäs", "hund"]), Chip(label="Tomt", words=["xyz"])]
    assert resolve(chips, VOCABULARY) == [{"label": "Maten", "words": [WORDS[0], WORDS[3]]}]


def test_a_chip_offers_a_sign_once_under_the_word_it_was_asked_for():
    assert resolve([Chip(label="Färger", words=["blå", "öga"])], VOCABULARY) == [{"label": "Färger", "words": [WORDS[1]]}]  # fmt: skip


def test_a_model_that_does_not_answer_offers_no_chips():
    def parse(**_):
        raise anthropic.APIConnectionError(request=None)

    assert suggest(SimpleNamespace(messages=SimpleNamespace(parse=parse)), VOCABULARY, "ny", "", []) == []
    answered = SimpleNamespace(parsed_output=Suggested(chips=[Chip(label="Djur", words=["hund"])]))
    client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **_: answered))
    assert suggest(client, VOCABULARY, "ny", "", []) == [{"label": "Djur", "words": [WORDS[3]]}]
