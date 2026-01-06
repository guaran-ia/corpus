"""
Sentence-level and document-level heuristic metrics.

All metrics in this module are general and corpus-agnostic.
They operate on plain text strings and can be applied to any corpus
from an external runner.
"""

import spacy
import re
from typing import List


# ---------------------------------------------------------------------
# NLP INITIALIZATION
# ---------------------------------------------------------------------

# Minimal multilingual pipeline for sentence segmentation and tokenization
_nlp = spacy.blank("xx")
_nlp.max_length=3000000
_nlp.add_pipe("sentencizer")


# ---------------------------------------------------------------------
# HELPER FUNCTIONS (SENTENCE LEVEL)
# ---------------------------------------------------------------------

def uppercase_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isupper())


def numbers_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isdigit())


def characters_count(sentence: str) -> int:
    return len(sentence)


def alphanumeric_count(sentence: str) -> int:
    return sum(1 for ch in sentence if ch.isalnum())


def words(sentence: str) -> List[str]:
    doc = _nlp(sentence)
    return [tok.text for tok in doc if tok.is_alpha]


def symbols_count(sentence: str) -> int:
    return sum(1 for ch in sentence if not ch.isalnum() and not ch.isspace())


def stopword_counts(sentence: str) -> tuple[int, int]:
    doc = _nlp(sentence)
    stop = sum(1 for tok in doc if tok.is_stop)
    non_stop = sum(1 for tok in doc if tok.is_alpha and not tok.is_stop)
    return stop, non_stop


def character_repetition_ratio(sentence: str) -> float:
    chars = [ch for ch in sentence if not ch.isspace()]
    if not chars:
        return 0.0
    return 1.0 - (len(set(chars)) / len(chars))


def word_repetition_ratio(sentence: str) -> float:
    ws = words(sentence)
    if not ws:
        return 0.0
    return 1.0 - (len(set(ws)) / len(ws))


# ---------------------------------------------------------------------
# HELPER FUNCTIONS (TEXT LEVEL)
# ---------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    doc = _nlp(text)
    return [sent.text for sent in doc.sents]


# ---------------------------------------------------------------------
# SENTENCE-LEVEL METRICS (AVERAGES / MIN / MAX)
# ---------------------------------------------------------------------

def average_uppercase_letters_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(uppercase_count(s) for s in sents) / len(sents)


def average_numbers_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(numbers_count(s) for s in sents) / len(sents)


def average_words_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(len(words(s)) for s in sents) / len(sents)


def average_characters_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(characters_count(s) for s in sents) / len(sents)


def average_alphanumeric_characters_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(alphanumeric_count(s) for s in sents) / len(sents)


def mean_word_length(text: str) -> float:
    doc = _nlp(text)
    ws = [tok.text for tok in doc if tok.is_alpha]
    if not ws:
        return 0.0
    return sum(len(w) for w in ws) / len(ws)


def average_sentence_length(text: str) -> float:
    return average_words_per_sentence(text)


def max_sentence_length(text: str) -> int:
    sents = sentences(text)
    if not sents:
        return 0
    return max(len(words(s)) for s in sents)


def min_sentence_length(text: str) -> int:
    sents = sentences(text)
    if not sents:
        return 0
    return min(len(words(s)) for s in sents)


def average_ratio_of_symbols_to_words(text: str) -> float:
    sents = sentences(text)
    ratios = []
    for s in sents:
        w = len(words(s))
        if w > 0:
            ratios.append(symbols_count(s) / w)
    return sum(ratios) / len(ratios) if ratios else 0.0


def average_ratio_of_stopwords_to_non_stopwords(text: str) -> float:
    sents = sentences(text)
    ratios = []
    for s in sents:
        stop, non_stop = stopword_counts(s)
        if non_stop > 0:
            ratios.append(stop / non_stop)
    return sum(ratios) / len(ratios) if ratios else 0.0


def average_character_repetition_ratio_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(character_repetition_ratio(s) for s in sents) / len(sents)


def average_word_repetition_ratio_per_sentence(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(word_repetition_ratio(s) for s in sents) / len(sents)


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
# MANUAL TEST (DEVELOPMENT ONLY)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    sample = "Hola Mundo. Esto es UNA prueba. En 2024 adopté 1 más."
    print("Avg uppercase:", average_uppercase_letters_per_sentence(sample))
    print("Avg numbers:", average_numbers_per_sentence(sample))
    print("Avg words:", average_words_per_sentence(sample))
    print("Mean word length:", mean_word_length(sample))

