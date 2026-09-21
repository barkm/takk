from types import SimpleNamespace

from takk.story import Told, story_parts, write_story


class FakeClient:
    """A stand-in for the Anthropic client: answers with each of `answers` (a story's parts) in turn."""

    def __init__(self, *answers: list[str]):
        self.answers = list(answers)
        self.asked: list[list[dict]] = []

    @property
    def messages(self):
        return SimpleNamespace(parse=self.parse)

    def parse(self, model, max_tokens, system, messages, output_format):
        self.asked.append(messages)
        return SimpleNamespace(parsed_output=Told(parts=self.answers.pop(0)))


WORDS = ["mamma", "mjölk", "sova"]


def test_story_parts_reads_off_each_part_which_words_it_signs():
    parts = ["Mamma häller upp mjölk.", "Sedan ska du sova."]
    assert story_parts(parts, WORDS, 2) == [
        ("Mamma häller upp mjölk.", ["mamma", "mjölk"]),
        ("Sedan ska du sova.", ["sova"]),
    ]


def test_story_parts_refuses_a_story_that_cannot_be_practised():
    assert story_parts(["Mamma sover."], WORDS, 2) is None  # one part where two were asked for
    assert story_parts(["Mamma häller upp mjölk.", "Sedan är det natt."], WORDS, 2) is None  # a part signs nothing
    assert story_parts(["Mamma vill sova, mamma."], WORDS, 1) is None  # twice, so which one is signed?
    # an inflected word is no word of the story's: here nothing is left to sign, so the part is refused
    assert story_parts(["Hon drack mjölken."], WORDS, 1) is None


def test_write_story_returns_the_parts_with_the_words_each_one_uses():
    client = FakeClient(["Mamma har mjölk.", "Nu ska alla sova."])
    assert write_story(client, WORDS, 2, model="test") == [
        ("Mamma har mjölk.", ["mamma", "mjölk"]),
        ("Nu ska alla sova.", ["sova"]),
    ]
    assert "2 delar" in client.asked[0][0]["content"]


def test_write_story_asks_again_when_a_part_cannot_be_scored():
    client = FakeClient(["Hon drack mjölken.", "Alla sover."], ["Mamma har mjölk.", "Nu ska alla sova."])
    assert write_story(client, WORDS, 2, model="test") is not None
    # the second ask carries the first answer and what was wrong with it, as `write_sentence` does
    assert [message["role"] for message in client.asked[1]] == ["user", "assistant", "user"]
    assert "mjölken" in client.asked[1][1]["content"]


def test_write_story_gives_up_rather_than_handing_back_a_story_it_cannot_time():
    client = FakeClient(["Hon drack mjölken."], ["Mjölken var god."])
    assert write_story(client, WORDS, 1, model="test") is None
