from takk.categories import sign_by_word, word


def test_word_is_the_label_without_its_entry_number():
    assert word("sts:platta slag-25563") == "platta slag"
    assert word("sts:0 - 0-01631") == "0 - 0"


def test_the_lowest_entry_number_wins_when_signs_share_a_word():
    signs = ["sts:Björn-00099", "sts:björn-00012", "sts:mjölk-01234"]
    assert sign_by_word(signs) == {"björn": "sts:björn-00012", "mjölk": "sts:mjölk-01234"}
