"""Small, deterministic text helpers for the scoring metrics.

Everything here is hand-auditable: token overlap, recall, number
extraction and proper-noun spotting. Nothing calls a model and
nothing is random, so two runs over the same inputs always agree.
"""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "i", "you", "we", "they", "he", "she", "my", "your",
    "our", "their", "his", "her", "do", "does", "did", "can", "could",
    "will", "would", "should", "may", "might", "please", "how", "what",
    "when", "where", "which", "who", "why", "there", "here", "from", "by",
    "at", "as", "an", "not", "no", "yes", "if", "so", "than", "then",
    "into", "over", "under", "per", "have", "has", "had", "am", "me",
}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def normalize(text: str) -> str:
    """Lower-case, fold accents, collapse whitespace."""
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", folded.lower()).strip()


def tokens(text: str) -> list[str]:
    """Word tokens from normalised text.

    Hyphens split words ("2-year" -> "2", "year") so a hyphenated
    form matches its plain form, decimals stay together ("6.95" is
    one token), and number words are folded to digits ("thirty" ->
    "30") so a spelled-out number matches the digit form.
    Apostrophes stay attached ("don't" stays one token).
    """
    raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?(?:'[a-z0-9]+)?", normalize(text))
    return [_NUMBER_WORDS.get(token, token) for token in raw]


def content_tokens(text: str) -> set[str]:
    """Tokens minus stopwords — the words that carry meaning."""
    return {t for t in tokens(text) if t not in _STOPWORDS}


def overlap(a: str, b: str) -> float:
    """Jaccard-style overlap of content tokens, 0.0–1.0."""
    set_a, set_b = content_tokens(a), content_tokens(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def recall(reference: str, response: str) -> float:
    """How much of the reference's content the response covers."""
    ref, resp = content_tokens(reference), content_tokens(response)
    if not ref:
        return 0.0
    return len(ref & resp) / len(ref)


def numbers(text: str) -> set[str]:
    """Numbers mentioned in the text: digits and number words.

    Currency and units are stripped, so "$6.95" and "6.95" match,
    and "thirty" matches "30".
    """
    found: set[str] = set()
    for token in tokens(text):
        cleaned = token.replace(",", "").strip("$£€")
        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            found.add(cleaned)
        elif cleaned in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[cleaned])
    return found


def proper_nouns(text: str) -> set[str]:
    """Capitalised words that are not sentence-initial.

    Used to spot entity substitution: a response naming a different
    proper noun in place of the reference's is contradicting it,
    even if it shares many ordinary words.
    """
    nouns: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        for word in sentence.split()[1:]:
            clean = re.sub(r"[^A-Za-zÀ-ÿ]", "", word)
            if len(clean) >= 3 and clean[0].isupper():
                nouns.add(clean.lower())
    return nouns


def sentences(text: str) -> list[str]:
    """Split text into sentences (naive, deterministic)."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return parts


def snippet(text: str, limit: int = 120) -> str:
    """First `limit` characters of a text, with an ellipsis if cut."""
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def clamp_score(value: float) -> int:
    """Clamp a raw value to the 1–5 rubric scale as an int."""
    return max(1, min(5, int(round(value))))
