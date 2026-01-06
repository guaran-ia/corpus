"""
Heuristics runner.

This module applies sentence-level and document-level heuristic metrics
to corpora stored as JSONL files in data/raw.

The runner does not store results yet; it only computes and prints them.
"""

import json
from pathlib import Path
from typing import List

from .sentence_metrics import (
    average_uppercase_letters_per_sentence,
    average_numbers_per_sentence,
    average_words_per_sentence,
    average_characters_per_sentence,
    average_alphanumeric_characters_per_sentence,
    mean_word_length,
    average_sentence_length,
    max_sentence_length,
    min_sentence_length,
    average_ratio_of_symbols_to_words,
    average_ratio_of_stopwords_to_non_stopwords,
    average_character_repetition_ratio_per_sentence,
    average_word_repetition_ratio_per_sentence,
    average_sentences_per_document,
    min_sentences_per_document,
    max_sentences_per_document,
)

RAW_DATA_DIR = Path("data/raw")
TEXT_FIELD = "text"


def read_jsonl_texts(path: Path) -> List[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if TEXT_FIELD in obj and obj[TEXT_FIELD].strip():
                texts.append(obj[TEXT_FIELD])
    return texts


def run_on_file(jsonl_path: Path) -> None:
    print(f"\nProcessing corpus file: {jsonl_path.name}")

    documents = read_jsonl_texts(jsonl_path)

    if not documents:
        print("No documents found.")
        return

    sentence_metrics = {
        "avg_uppercase_letters_per_sentence":
            sum(average_uppercase_letters_per_sentence(t) for t in documents) / len(documents),

        "avg_numbers_per_sentence":
            sum(average_numbers_per_sentence(t) for t in documents) / len(documents),

        "avg_words_per_sentence":
            sum(average_words_per_sentence(t) for t in documents) / len(documents),

        "avg_characters_per_sentence":
            sum(average_characters_per_sentence(t) for t in documents) / len(documents),

        "avg_alphanumeric_characters_per_sentence":
            sum(average_alphanumeric_characters_per_sentence(t) for t in documents) / len(documents),

        "mean_word_length":
            sum(mean_word_length(t) for t in documents) / len(documents),

        "avg_sentence_length":
            sum(average_sentence_length(t) for t in documents) / len(documents),

        "max_sentence_length":
            max(max_sentence_length(t) for t in documents),

        "min_sentence_length":
            min(min_sentence_length(t) for t in documents),

        "avg_symbol_to_word_ratio":
            sum(average_ratio_of_symbols_to_words(t) for t in documents) / len(documents),

        "avg_stopword_to_non_stopword_ratio":
            sum(average_ratio_of_stopwords_to_non_stopwords(t) for t in documents) / len(documents),

        "avg_character_repetition_ratio":
            sum(average_character_repetition_ratio_per_sentence(t) for t in documents) / len(documents),

        "avg_word_repetition_ratio":
            sum(average_word_repetition_ratio_per_sentence(t) for t in documents) / len(documents),
    }

    document_metrics = {
        "avg_sentences_per_document":
            average_sentences_per_document(documents),

        "min_sentences_per_document":
            min_sentences_per_document(documents),

        "max_sentences_per_document":
            max_sentences_per_document(documents),
    }

    print("Sentence-level metrics:")
    for k, v in sentence_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Document-level metrics:")
    for k, v in document_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":

    if not RAW_DATA_DIR.exists():
        print("data/raw directory not found.")
        raise SystemExit(1)

    for jsonl_file in sorted(RAW_DATA_DIR.glob("*.jsonl")):
        run_on_file(jsonl_file)

