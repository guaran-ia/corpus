"""
Sentence-level and document-level heuristic metrics.
"""
from .utils import normalize_guarani
import unicodedata
import re
from typing import List
from pathlib import Path
import spacy
from src.pipeline.tokenization import tokenize



# ---------------------------------------------------------------------
# NLP INITIALIZATION (sentence segmentation only)
# ---------------------------------------------------------------------

_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 10_000_000


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
# BASIC TEXT HELPERS
# ---------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    doc = _nlp(text)
    return [sent.text for sent in doc.sents]


def words(text: str) -> List[str]:
    return list(tokenize(text))


# ---------------------------------------------------------------------
# BAD WORDS (loaded directly from txt)
# ---------------------------------------------------------------------

BAD_WORDS_PATH = Path(__file__).parent / "bad_words.txt"


def load_bad_words() -> list[str]:
    with BAD_WORDS_PATH.open(encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


BAD_WORDS = load_bad_words()


# ---------------------------------------------------------------------
# LANGUAGE IDENTIFICATION (Guarani proportion)
# ---------------------------------------------------------------------

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
    Return the number of stopwords in a sentence.
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
    total_symbols = 0
    total_words = 0

    for s in sentences(text):
        total_symbols += symbols_count(s)
        total_words += len(words(s))

    return total_symbols / total_words if total_words else 0.0


def average_ratio_of_stopwords_to_non_stopwords(text: str) -> float:
    total_stopwords = 0
    total_words = 0

    for s in sentences(text):
        tokens = words(s)
        total_words += len(tokens)
        total_stopwords += stopword_count(s)

    non_stopwords = total_words - total_stopwords
    return total_stopwords / non_stopwords if non_stopwords else 0.0


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


# ---------------------------------------------------------------------
# HEURISTIC COUNTS AND FLAGS
# ---------------------------------------------------------------------

def count_lorem_ipsum_sentences(text: str) -> int:
    return sum(
        1
        for sentence in sentences(text)
        if "lorem ipsum" in sentence.lower()
    )


def count_sentences_ending_with_ellipsis(text: str) -> int:
    return sum(
        1
        for sentence in sentences(text)
        if sentence.rstrip().endswith("...")
    )


def count_sentences_starting_with_bullet(text: str) -> int:
    """
    Count the number of sentences that start with a bullet point.
    Includes hyphen, en dash, and em dash.
    """
    bullet_chars = ("-", "•", "*", "–", "—")

    return sum(
        1
        for sentence in sentences(text)
        if sentence.lstrip().startswith(bullet_chars)
    )


def count_sentences_without_terminal_punctuation(text: str) -> int:
    terminal_punctuation = (".", "!", "?", "\"", "”", "’", "»")

    count = 0
    for sentence in sentences(text):
        s = sentence.rstrip()
        if s and not s.endswith(terminal_punctuation):
            count += 1

    return count


def count_sentences_with_curly_bracket(text: str) -> int:
    return sum(
        1
        for sentence in sentences(text)
        if "{" in sentence
    )
 
    """
    Includes possible list of guarani legal phrases.
   
     """

LEGAL_PHRASES = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
    "términos de uso",
    "politica de privacidad",
    "política de privacidad",
    "politica de cookies",
    "política de cookies",
    "usa cookies",
    "uso de cookies",
    "cookies jeporu", 
    "ñe’ẽme’ẽ jepururã",
    "cookies rehegua marandu",
    "oipuru cookies",
    "cookies jeporu",
    "ojepuru cookies",

]


def count_sentences_with_legal_phrases(text: str) -> int:
    count = 0
    for sentence in sentences(text):
        s = sentence.lower()
        if any(phrase in s for phrase in LEGAL_PHRASES):
            count += 1
    return count


def count_sentences_with_javascript(text: str) -> int:
    return sum(
        1
        for sentence in sentences(text)
        if "javascript" in sentence.lower()
    )


def average_words_in_sentences_starting_with_capital(text: str) -> float:
    """
    Compute the average number of words in sentences that contain
    at least one word starting with an uppercase letter.
    """
    word_counts = []

    for sentence in sentences(text):
        ws = words(sentence)
        if not ws:
            continue

        has_capitalized_word = any(
            w and w[0].isupper()
            for w in ws
        )

        if has_capitalized_word:
            word_counts.append(len(ws))

    return sum(word_counts) / len(word_counts) if word_counts else 0.0



def count_bad_words_occurrences(text: str) -> int:
    """
    Count the number of occurrences of bad words or phrases
    using orthographic normalization for Guaraní.
    """
    normalized_text = normalize_guarani(text)
    count = 0

    for phrase in BAD_WORDS:
        normalized_phrase = normalize_guarani(phrase)
        count += normalized_text.count(normalized_phrase)

    return count



def count_sentences_with_low_guarani_proportion(
    text: str,
    threshold: float = 0.7,
) -> int:
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
