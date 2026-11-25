"""Tests for the deterministic text helpers."""

from northstar_eval.text import (
    clamp_score,
    content_tokens,
    normalize,
    numbers,
    overlap,
    proper_nouns,
    recall,
    sentences,
    snippet,
    tokens,
)


def test_normalize_folds_accents_and_case():
    assert normalize("  Café  NOIR ") == "cafe noir"


def test_tokens_split_hyphens_and_fold_number_words():
    assert tokens("2-year warranty, thirty days") == ["2", "year", "warranty", "30", "days"]


def test_tokens_keep_apostrophes():
    assert "don't" in tokens("I don't know")


def test_content_tokens_drop_stopwords():
    assert content_tokens("How long do I have to return a jacket?") == {
        "long", "return", "jacket",
    }


def test_overlap_identical_is_one():
    assert overlap("red tent blue", "red tent blue") == 1.0


def test_overlap_disjoint_is_zero():
    assert overlap("red tent", "blue jacket") == 0.0


def test_recall_full_and_empty():
    assert recall("red tent blue", "the red tent is blue") == 1.0
    assert recall("red tent blue", "green jacket") == 0.0
    assert recall("red tent", "") == 0.0


def test_numbers_digits_currency_and_words():
    assert numbers("Standard shipping is $6.95 for orders over $75") == {"6.95", "75"}
    assert numbers("thirty days") == {"30"}
    assert numbers("2-year warranty") == {"2"}


def test_proper_nouns_spot_substitution():
    assert proper_nouns("The capital of France is Paris.") == {"paris", "france"}
    assert proper_nouns("Paris is lovely.") == set()  # Sentence-initial is not counted.


def test_sentences_split():
    assert len(sentences("One. Two! Three?")) == 3
    assert sentences("") == []


def test_snippet_truncates():
    assert snippet("x" * 200, limit=10).endswith("…")
    assert snippet("short") == "short"


def test_clamp_score_bounds():
    assert clamp_score(0) == 1
    assert clamp_score(9) == 5
    assert clamp_score(3.4) == 3
