from types import SimpleNamespace

import anthropic

from takk.story import Part, Signed, Told, signed_words, story_parts, write_story


def part(text: str, *signs: tuple[str, str]) -> Part:
    return Part(text=text, signs=[Signed(said=said, word=word) for said, word in signs])


class FakeClient:
    """A stand-in for the Anthropic client: answers with each of `answers` (a story's parts) in turn.
    An answer of None is a response with nothing parsed, as a turn cut off by `max_tokens` is, and an
    exception is raised where the API would raise it."""

    def __init__(self, *answers: list[Part] | None | Exception):
        self.answers = list(answers)
        self.asked: list[list[dict]] = []

    @property
    def messages(self):
        return SimpleNamespace(parse=self.parse)

    def parse(self, model, max_tokens, system, messages, output_config, output_format):
        self.asked.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(parsed_output=Told(parts=answer) if answer is not None else None)


WORDS = ["mamma", "mjölk", "sova"]


def test_signed_words_takes_the_forms_the_part_actually_says():
    # the point of reporting rather than reading the words off the text: the form is inflected freely
    assert signed_words(part("Mamma ger mjölken.", ("Mamma", "mamma"), ("mjölken", "mjölk")), WORDS)
    assert signed_words(part("Barnet sov gott.", ("sov", "sova")), WORDS)
    assert signed_words(part("Vi äter platta slag.", ("platta slag", "platta slag")), ["platta slag"])


def test_signed_words_refuses_what_it_can_check():
    # a form the part does not say at all
    assert signed_words(part("Mamma ger mjölk.", ("bilen", "bil")), [*WORDS, "bil"]) is None
    # a word that was never offered
    assert signed_words(part("Pappa ger mjölk.", ("Pappa", "pappa")), WORDS) is None
    # signs out of the order they are spoken, which is the order the audio is cut in
    assert signed_words(part("Mamma ger mjölk.", ("mjölk", "mjölk"), ("Mamma", "mamma")), WORDS) is None
    # a part that signs nothing at all
    assert signed_words(part("Det var en gång."), WORDS) is None
    # more signs than one recording can carry
    many = part("Mamma ger mjölk och vill sova.", ("Mamma", "mamma"), ("mjölk", "mjölk"), ("sova", "sova"))  # fmt: skip
    assert signed_words(many, WORDS) is not None and signed_words(many, WORDS, most=2) is None


def test_signed_words_finds_a_word_said_twice_twice():
    """A word said twice is signed twice: alignment is positional, so two occurrences are two spans,
    and the second `mjölk` is looked for after the first rather than on top of it."""
    twice = part("Mamma ger mjölk, och sedan mer mjölk.", ("mjölk", "mjölk"), ("mjölk", "mjölk"))
    assert signed_words(twice, WORDS) is not None
    # said once, so the second sign has nothing left to be
    assert signed_words(part("Mamma ger mjölk.", ("mjölk", "mjölk"), ("mjölk", "mjölk")), WORDS) is None


def test_story_parts_accepts_a_story_that_is_not_the_length_it_was_asked_for():
    """How many parts there are is what the story aims at, not a contract: one part fewer is a
    shorter pass rather than a story to throw away."""
    assert story_parts([part("Mamma ger mjölk.", ("Mamma", "mamma"))], WORDS) is not None
    assert story_parts([], WORDS) is None
    # but every part has to be one the learner can sign
    assert story_parts([part("Mamma ger mjölk.", ("Mamma", "mamma")), part("Det var kväll.")], WORDS) is None  # fmt: skip


def test_write_story_returns_the_parts_with_the_signs_each_one_is_practised_by():
    told = [part("Mamma har mjölken.", ("Mamma", "mamma"), ("mjölken", "mjölk")), part("Nu ska alla sova.", ("sova", "sova"))]  # fmt: skip
    client = FakeClient(told)
    story = write_story(client, WORDS, 2, model="test")
    assert story is not None
    assert [(sign.said, sign.word) for sign in story[0].signs] == [("Mamma", "mamma"), ("mjölken", "mjölk")]  # fmt: skip
    assert "2 delar" in client.asked[0][0]["content"]


def test_write_story_asks_again_when_a_part_cannot_be_scored():
    # the reported form is not in the part at all, which is the half of the contract that is checked
    refused = [part("Hon drack upp allt.", ("mjölken", "mjölk")), part("Alla sover.", ("sover", "sova"))]
    written = [part("Mamma har mjölk.", ("mjölk", "mjölk")), part("Nu ska alla sova.", ("sova", "sova"))]
    client = FakeClient(refused, written)
    assert write_story(client, WORDS, 2, model="test") is not None
    # the second ask carries the first answer and what was wrong with it
    assert [message["role"] for message in client.asked[1]] == ["user", "assistant", "user"]
    assert "Hon drack upp allt." in client.asked[1][1]["content"]


def test_write_story_asks_again_when_the_answer_was_cut_short():
    written = [part("Mamma har mjölk.", ("mjölk", "mjölk"))]
    assert write_story(FakeClient(None, written), WORDS, 2, model="test") is not None
    assert write_story(FakeClient(None), WORDS, 2, model="test", tries=1) is None


def test_write_story_asks_again_when_the_api_could_not_answer():
    # an overloaded or rate limited API is a try that failed, not a 500 at the learner
    failed = anthropic.APIConnectionError(request=SimpleNamespace())
    written = [part("Mamma har mjölk.", ("mjölk", "mjölk"))]
    assert write_story(FakeClient(failed, written), WORDS, 2, model="test") is not None
    assert write_story(FakeClient(failed), WORDS, 2, model="test", tries=1) is None


def test_write_story_gives_up_rather_than_handing_back_a_story_it_cannot_score():
    refused = [part("Hon drack upp allt.", ("mjölken", "mjölk"))]
    assert write_story(FakeClient(refused, refused), WORDS, 1, model="test") is None
