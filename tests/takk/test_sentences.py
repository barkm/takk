from types import SimpleNamespace

from takk.sentences import Written, key_words, write_sentence


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


def test_key_words_reads_off_the_sentence_which_offered_words_it_used():
    assert key_words("Jag vill ha mer mjölk", ["mjölk", "mer"]) == ["mer", "mjölk"]
    assert key_words("Mjölk är gott", ["mjölk"]) == ["mjölk"]  # a sentence-initial capital still matches
    assert key_words("Vi ska äta platta slag idag", ["äta", "platta slag"]) == ["äta", "platta slag"]  # a sign of two words
    # the words after the first are offered, not required, so a sentence may leave them out
    assert key_words("Jag vill ha mjölk", ["mjölk", "mer", "bröd"]) == ["mjölk"]
    assert key_words("Jag vill ha mer mjölk", ["mjölk", "mer", "bröd"]) == ["mer", "mjölk"]
    assert key_words("Vi plockar blåbär", ["blåbär", "blå"]) == ["blåbär"]  # "blå" is not found inside "blåbär"


def test_key_words_refuses_a_sentence_it_could_not_be_scored_from():
    assert key_words("Jag vill ha mera mjölk", ["mer", "mjölk"]) is None  # "mera" is not "mer", so the head is missing
    assert key_words("Jag drack mjölken", ["mjölk"]) is None  # inflected, so it would not be timed in the audio
    assert key_words("Mjölk är gott", ["mer", "mjölk"]) is None  # the head is the word the turn is for
    assert key_words("Mer mjölk och mer bröd", ["mer", "bröd"]) is None  # twice, so which one is signed?
    # at most three signs in one recording, however many words were offered
    assert key_words("Mamma vill ha mer mjölk och bröd", ["mer", "mjölk", "bröd", "mamma"]) is None


def test_write_sentence_returns_the_sentence_and_the_words_it_chose_to_use():
    client = FakeClient("Jag vill ha mer mjölk")
    # bröd was offered and left out, which is the model's choice to make; mer and mjölk come back in
    # the order they are spoken, which is the order they are signed in
    assert write_sentence(client, ["mjölk", "mer", "bröd"], model="test") == ("Jag vill ha mer mjölk", ["mer", "mjölk"])
    assert "mjölk" in client.asked[0][0]["content"]  # the head is asked for by name, the rest offered


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
