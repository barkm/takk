from types import SimpleNamespace

from takk.sentences import Written, key_order, write_sentence


class FakeClient:
    """A stand-in for the Anthropic client: answers with each of `answers` in turn."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.asked: list[list[dict]] = []

    @property
    def messages(self):
        return SimpleNamespace(parse=self.parse)

    def parse(self, model, max_tokens, system, messages, output_format):
        self.asked.append(messages)
        return SimpleNamespace(parsed_output=Written(sentence=self.answers.pop(0)))


def test_key_order_reads_the_words_off_the_sentence():
    assert key_order("Jag vill ha mer mjölk", ["mjölk", "mer"]) == ["mer", "mjölk"]
    assert key_order("Mjölk är gott", ["mjölk"]) == ["mjölk"]  # a sentence-initial capital still matches
    assert key_order("Vi ska äta platta slag idag", ["äta", "platta slag"]) == ["äta", "platta slag"]  # a sign of two words


def test_key_order_refuses_a_sentence_that_does_not_use_each_word_once_as_written():
    assert key_order("Jag vill ha mera mjölk", ["mer", "mjölk"]) is None  # "mera" is not "mer"
    assert key_order("Jag drack mjölken", ["mjölk"]) is None  # inflected, so it would not be timed in the audio
    assert key_order("Jag vill ha mjölk", ["mjölk", "mer"]) is None  # a word left out
    assert key_order("Mer mjölk och mer bröd", ["mer", "bröd"]) is None  # twice, so which one is signed?


def test_write_sentence_returns_the_sentence_and_the_order_its_words_come_in():
    client = FakeClient("Jag vill ha mer mjölk")
    assert write_sentence(client, ["mjölk", "mer"], model="test") == ("Jag vill ha mer mjölk", ["mer", "mjölk"])


def test_write_sentence_asks_again_when_a_word_was_inflected():
    client = FakeClient("Jag drack mjölken", "Jag vill ha mjölk")
    assert write_sentence(client, ["mjölk"], model="test") == ("Jag vill ha mjölk", ["mjölk"])
    # the second ask carries the first answer and what was wrong with it, not the bare question again
    assert [message["role"] for message in client.asked[1]] == ["user", "assistant", "user"]
    assert "mjölken" in client.asked[1][1]["content"]


def test_write_sentence_gives_up_rather_than_handing_back_a_sentence_it_cannot_time():
    """The caller then practises the words one at a time, which needs no sentence at all."""
    client = FakeClient("Jag drack mjölken", "Mjölken var god")
    assert write_sentence(client, ["mjölk"], model="test") is None
