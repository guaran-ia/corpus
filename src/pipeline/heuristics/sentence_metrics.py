"""
Sentence-level and document-level heuristic metrics.
"""

from typing import List
from pathlib import Path
import spacy
from .tokenization import tokenize

# ---------------------------------------------------------------------
# NLP INITIALIZATION (sentence segmentation only)
# ---------------------------------------------------------------------

_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 3_000_000


# ---------------------------------------------------------------------
# STOPWORDS (loaded directly from txt)
# ---------------------------------------------------------------------

STOPWORDS_PATH = Path(__file__).parent / "stopwords.txt"

def load_stopwords() -> set[str]:
    with STOPWORDS_PATH.open(encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }

STOPWORDS = load_stopwords()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    doc = _nlp(text)
    return [sent.text for sent in doc.sents]


def words(text: str) -> List[str]:
    return list(tokenize(text))
BAD_WORDS_PATH = Path(__file__).parent / "bad_words.txt"


def load_bad_words() -> list[str]:
    with BAD_WORDS_PATH.open(encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


BAD_WORDS = load_bad_words()

# sentence_metrics.py (arriba del archivo)

try:
    from src.pipeline.language_identifier.language_identifier import LanguageIdentifier

    _LANGID = LanguageIdentifier(
        fasttext=True,
        glotlid=True,
        openlid=True,
    )
except ImportError:
    _LANGID = None

GUARANI_CODES = {"gn", "gug", "grn"}

# ---------------------------------------------------------------------
# CHARACTER-LEVEL HELPERS
# ---------------------------------------------------------------------

def uppercase_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isupper())


def numbers_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isdigit())


def characters_count(sentence: str) -> int:
    return len(sentence)


def alphanumeric_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isalnum())


def symbols_count(sentence: str) -> int:
    return sum(1 for ch in sentence if not ch.isalnum() and not ch.isspace())


# ---------------------------------------------------------------------
# WORD-LEVEL HELPERS
# ---------------------------------------------------------------------

def stopword_count(sentence: str) -> int:
    """
    Returns the number of stopwords in a sentence.
    """
    count = 0
    for tok in tokenize(sentence):
        if tok in STOPWORDS:
            count += 1
    return count



def word_repetition_ratio(sentence: str) -> float:
    ws = words(sentence)
    if not ws:
        return 0.0
    return 1.0 - (len(set(ws)) / len(ws))


def character_repetition_ratio(sentence: str) -> float:
    chars = [ch for ch in sentence if not ch.isspace()]
    if not chars:
        return 0.0
    return 1.0 - (len(set(chars)) / len(chars))


# ---------------------------------------------------------------------
# SENTENCE-LEVEL METRICS
# ---------------------------------------------------------------------

def average_uppercase_letters_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(uppercase_count(s) for s in sents) / len(sents) if sents else 0.0


def average_numbers_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(numbers_count(s) for s in sents) / len(sents) if sents else 0.0


def average_words_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(len(words(s)) for s in sents) / len(sents) if sents else 0.0

def average_sentence_length(text: str) -> float:
    """
    Alias of average_words_per_sentence.
    Sentence length measured in number of words.
    """
    return average_words_per_sentence(text)

def average_characters_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(characters_count(s) for s in sents) / len(sents) if sents else 0.0


def average_alphanumeric_characters_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(alphanumeric_count(s) for s in sents) / len(sents) if sents else 0.0


def mean_word_length(text: str) -> float:
    ws = []
    for s in sentences(text):
        ws.extend(words(s))
    return sum(len(w) for w in ws) / len(ws) if ws else 0.0


def max_sentence_length(text: str) -> int:
    sents = sentences(text)
    return max(len(words(s)) for s in sents) if sents else 0


def min_sentence_length(text: str) -> int:
    sents = sentences(text)
    return min(len(words(s)) for s in sents) if sents else 0


def average_ratio_of_symbols_to_words(text: str) -> float:
    """
    Computes the ratio of symbols to words over the entire text.
    """
    total_symbols = 0
    total_words = 0

    for s in sentences(text):
        total_symbols += symbols_count(s)
        total_words += len(words(s))

    if total_words == 0:
        return 0.0

    return total_symbols / total_words



def average_ratio_of_stopwords_to_non_stopwords(text: str) -> float:
    """
    Computes the ratio of stopwords to non-stopwords over the entire text.
    """
    total_stopwords = 0
    total_words = 0

    for s in sentences(text):
        tokens = words(s)
        total_words += len(tokens)
        total_stopwords += stopword_count(s)

    non_stopwords = total_words - total_stopwords
    if non_stopwords == 0:
        return 0.0

    return total_stopwords / non_stopwords



def average_character_repetition_ratio_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(character_repetition_ratio(s) for s in sents) / len(sents) if sents else 0.0


def average_word_repetition_ratio_per_sentence(text: str) -> float:
    sents = sentences(text)
    return sum(word_repetition_ratio(s) for s in sents) / len(sents) if sents else 0.0


# ---------------------------------------------------------------------
# DOCUMENT-LEVEL METRICS
# ---------------------------------------------------------------------

def sentence_counts_per_document(texts: List[str]) -> List[int]:
    return [len(sentences(t)) for t in texts]


def average_sentences_per_document(texts: List[str]) -> float:
    counts = sentence_counts_per_document(texts)
    return sum(counts) / len(counts) if counts else 0.0


def min_sentences_per_document(texts: List[str]) -> int:
    counts = sentence_counts_per_document(texts)
    return min(counts) if counts else 0


def max_sentences_per_document(texts: List[str]) -> int:
    counts = sentence_counts_per_document(texts)
    return max(counts) if counts else 0

# --------------------------------------------------
# New heuristic
# --------------------------------------------------

def count_lorem_ipsum_sentences(text: str) -> int:
    """
    Count the number of sentences that contain the placeholder
    phrase 'lorem ipsum'.
    """
    return sum(
        1
        for sentence in sentences(text)
        if "lorem ipsum" in sentence.lower()
    )
def count_sentences_ending_with_ellipsis(text: str) -> int:
    """
    Count the number of sentences that end with ellipsis (...).
    """
    return sum(
        1
        for sentence in sentences(text)
        if sentence.rstrip().endswith("...")
    )
def count_sentences_starting_with_bullet(text: str) -> int:
    """
    Count the number of sentences that start with a bullet point.
    """
    bullet_chars = ("-", "•", "*", "–", "—")

    return sum(
        1
        for sentence in sentences(text)
        if sentence.lstrip().startswith(bullet_chars)
    )
def count_sentences_without_terminal_punctuation(text: str) -> int:
    """
    Count sentences that do not end with terminal punctuation.
    """
    terminal_punctuation = (".", "!", "?", "\"", "”", "’", "»")

    count = 0
    for sentence in sentences(text):
        s = sentence.rstrip()
        if not s:
            continue

        if not s.endswith(terminal_punctuation):
            count += 1

    return count
def count_sentences_with_curly_bracket(text: str) -> int:
    """
    Count sentences that contain a curly bracket '{',
    which may indicate programming source code.
    """
    count = 0
    for sentence in sentences(text):
        if "{" in sentence:
            count += 1
    return count

LEGAL_PHRASES = [
    # English
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",

    # Spanish
    "términos de uso",
    "politica de privacidad",
    "política de privacidad",
    "politica de cookies",
    "política de cookies",
    "usa cookies",
    "uso de cookies",
]

def count_sentences_with_legal_phrases(text: str) -> int:
    """
    Count sentences that contain legal or cookie-related phrases.
    """
    count = 0
    for sentence in sentences(text):
        s = sentence.lower()
        if any(phrase in s for phrase in LEGAL_PHRASES):
            count += 1
    return count




def count_sentences_with_javascript(text: str) -> int:
    """
    Count sentences that contain the word 'JavaScript' or 'Javascript'.
    """
    count = 0

    for sentence in sentences(text):
        s = sentence.lower()
        if "javascript" in s:
            count += 1

    return count
def average_words_in_sentences_starting_with_capital(text: str) -> float:
    """
    Compute the average number of words in sentences that start
    with a capital letter.
    """
    word_counts = []

    for sentence in sentences(text):
        stripped = sentence.lstrip()

        if not stripped:
            continue

        first_char = stripped[0]

        if first_char.isupper():
            words_in_sentence = len(words(sentence))
            if words_in_sentence > 0:
                word_counts.append(words_in_sentence)

    return sum(word_counts) / len(word_counts) if word_counts else 0.0
def count_bad_words_occurrences(text: str) -> int:
    """
    Count the number of occurrences of bad words/phrases
    in a document, based on a predefined list.
    """
    text_lower = text.lower()
    count = 0

    for phrase in BAD_WORDS:
        count += text_lower.count(phrase)

    return count
def count_sentences_with_low_guarani_proportion(
    text: str,
    threshold: float = 0.7,
) -> int:
    """
    Count sentences whose proportion of Guarani is below a given threshold.
    """
    if _LANGID is None:
        return 0

    count = 0

    for sentence in sentences(text):
        result = _LANGID.predict(sentence)

        guarani_score = sum(
            score
            for lang, score in result.items()
            if lang in GUARANI_CODES
        )

        if guarani_score < threshold:
            count += 1

    return count
