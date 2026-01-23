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

def stopword_counts(sentence: str) -> tuple[int, int]:
    stop = 0
    non_stop = 0

    for tok in tokenize(sentence):
        if tok in STOPWORDS:
            stop += 1
        else:
            non_stop += 1

    return stop, non_stop


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
    ratios = []
    for s in sentences(text):
        w = len(words(s))
        if w > 0:
            ratios.append(symbols_count(s) / w)
    return sum(ratios) / len(ratios) if ratios else 0.0


def average_ratio_of_stopwords_to_non_stopwords(text: str) -> float:
    ratios = []
    for s in sentences(text):
        stop, non_stop = stopword_counts(s)
        if non_stop > 0:
            ratios.append(stop / non_stop)
    return sum(ratios) / len(ratios) if ratios else 0.0


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

