import json
from pathlib import Path

from .sentence_metrics import (
    average_uppercase_letters_per_sentence,
    average_numbers_per_sentence,
    average_words_per_sentence,
    average_characters_per_sentence,
    average_alphanumeric_characters_per_sentence,
    mean_word_length,
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

# ---------------------------------------------------------------------
# DATA DIRECTORY
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def read_jsonl(path: Path):
    """Yield JSON records from a .jsonl file."""
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def extract_text(record: dict) -> str:
    """
    Extract text from common fields used in GuaranIA corpora.
    """
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def run_on_file(path: Path) -> dict:
    """
    Run all heuristics on a single corpus file.
    """
    texts = []
    for record in read_jsonl(path):
        text = extract_text(record)
        if text:
            texts.append(text)

    if not texts:
        return {}

    # Sentence-level metrics (per document)
    sentence_metrics = {
        "avg_uppercase_letters_per_sentence": [
            average_uppercase_letters_per_sentence(t) for t in texts
        ],
        "avg_numbers_per_sentence": [
            average_numbers_per_sentence(t) for t in texts
        ],
        "avg_words_per_sentence": [
            average_words_per_sentence(t) for t in texts
        ],
        "avg_characters_per_sentence": [
            average_characters_per_sentence(t) for t in texts
        ],
        "avg_alphanumeric_characters_per_sentence": [
            average_alphanumeric_characters_per_sentence(t) for t in texts
        ],
        "mean_word_length": [
            mean_word_length(t) for t in texts
        ],
        "max_sentence_length": [
            max_sentence_length(t) for t in texts
        ],
        "min_sentence_length": [
            min_sentence_length(t) for t in texts
        ],
        "ratio_symbols_to_words": [
            average_ratio_of_symbols_to_words(t) for t in texts
        ],
        "ratio_stopwords_to_non_stopwords": [
            average_ratio_of_stopwords_to_non_stopwords(t) for t in texts
        ],
        "avg_character_repetition_ratio_per_sentence": [
            average_character_repetition_ratio_per_sentence(t) for t in texts
        ],
        "avg_word_repetition_ratio_per_sentence": [
            average_word_repetition_ratio_per_sentence(t) for t in texts
        ],
    }

    # Document-level metrics
    document_metrics = {
        "avg_sentences_per_document": average_sentences_per_document(texts),
        "min_sentences_per_document": min_sentences_per_document(texts),
        "max_sentences_per_document": max_sentences_per_document(texts),
    }

    return {
        "sentence_metrics": sentence_metrics,
        "document_metrics": document_metrics,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    for jsonl_file in DATA_DIR.glob("*.jsonl"):
        print(f"\nProcessing corpus: {jsonl_file.name}")
        results = run_on_file(jsonl_file)

        if not results:
            print("No valid texts found.")
            continue

        print("Sentence-level metrics (per document):")
        for k, v in results["sentence_metrics"].items():
            print(f"  {k}: {v}")

        print("Document-level metrics:")
        for k, v in results["document_metrics"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
