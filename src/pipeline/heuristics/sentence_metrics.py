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

