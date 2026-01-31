import json
from pathlib import Path

from .sentence_metrics import (
    # --------------------------------------------------
    # OLD METRICS (analysis only, corpus-level)
    # --------------------------------------------------
    average_uppercase_letters_per_sentence,
    average_numbers_per_sentence,
    average_words_per_sentence,
    average_characters_per_sentence,
    average_alphanumeric_characters_per_sentence,
    mean_word_length,
    max_sentence_length,
    min_sentence_length,
    average_sentence_length,
    average_ratio_of_symbols_to_words,
    average_ratio_of_stopwords_to_non_stopwords,
    average_character_repetition_ratio_per_sentence,
    average_word_repetition_ratio_per_sentence,

    # --------------------------------------------------
    # NEW METRICS (stored per document)
    # --------------------------------------------------
    count_lorem_ipsum_sentences,
    count_sentences_ending_with_ellipsis,
    count_sentences_starting_with_bullet,
    count_sentences_without_terminal_punctuation,
    count_sentences_with_curly_bracket,
    count_sentences_with_legal_phrases,
    count_sentences_with_low_guarani_proportion,
    count_sentences_with_javascript,
    average_words_in_sentences_starting_with_capital,
    count_bad_words_occurrences,
)

# ---------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def read_jsonl(path: Path):
    """Yield JSON records from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: Path, records):
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_text(record: dict) -> str:
    """
    Extract text from common fields used in GuaranIA corpora.
    """
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""

# ---------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path):
    """
    Process a single corpus file:
    - compute old metrics at corpus level (printed once)
    - compute new metrics per document (stored in processed JSONL)
    """
    processed_records = []

    # Accumulators for old metrics
    old_metrics = {
        "avg_uppercase_letters_per_sentence": [],
        "avg_numbers_per_sentence": [],
        "avg_words_per_sentence": [],
        "avg_characters_per_sentence": [],
        "avg_alphanumeric_characters_per_sentence": [],
        "mean_word_length": [],
        "max_sentence_length": [],
        "min_sentence_length": [],
        "avg_sentence_length": [],
        "ratio_symbols_to_words": [],
        "ratio_stopwords_to_non_stopwords": [],
        "avg_character_repetition_ratio": [],
        "avg_word_repetition_ratio": [],
    }

    print(f"\nProcessing corpus: {input_path.name}")

    for record in read_jsonl(input_path):
        text = extract_text(record)
        if not text:
            continue

        # --------------------------------------------------------------
        # OLD METRICS (accumulate only)
        # --------------------------------------------------------------
        old_metrics["avg_uppercase_letters_per_sentence"].append(
            average_uppercase_letters_per_sentence(text)
        )
        old_metrics["avg_numbers_per_sentence"].append(
            average_numbers_per_sentence(text)
        )
        old_metrics["avg_words_per_sentence"].append(
            average_words_per_sentence(text)
        )
        old_metrics["avg_characters_per_sentence"].append(
            average_characters_per_sentence(text)
        )
        old_metrics["avg_alphanumeric_characters_per_sentence"].append(
            average_alphanumeric_characters_per_sentence(text)
        )
        old_metrics["mean_word_length"].append(
            mean_word_length(text)
        )
        old_metrics["max_sentence_length"].append(
            max_sentence_length(text)
        )
        old_metrics["min_sentence_length"].append(
            min_sentence_length(text)
        )
        old_metrics["avg_sentence_length"].append(
            average_sentence_length(text)
        )
        old_metrics["ratio_symbols_to_words"].append(
            average_ratio_of_symbols_to_words(text)
        )
        old_metrics["ratio_stopwords_to_non_stopwords"].append(
            average_ratio_of_stopwords_to_non_stopwords(text)
        )
        old_metrics["avg_character_repetition_ratio"].append(
            average_character_repetition_ratio_per_sentence(text)
        )
        old_metrics["avg_word_repetition_ratio"].append(
            average_word_repetition_ratio_per_sentence(text)
        )

        # --------------------------------------------------------------
        # NEW METRICS (stored per document)
        # --------------------------------------------------------------
        record["heuristics"] = {
            "num_lorem_ipsum_sentences":
                count_lorem_ipsum_sentences(text),
            "num_sentences_ending_with_ellipsis":
                count_sentences_ending_with_ellipsis(text),
            "num_sentences_starting_with_bullet":
                count_sentences_starting_with_bullet(text),
            "num_sentences_without_terminal_punctuation":
                count_sentences_without_terminal_punctuation(text),
            "num_sentences_with_curly_bracket":
                count_sentences_with_curly_bracket(text),
            "num_sentences_with_legal_phrases":
                count_sentences_with_legal_phrases(text),
            "num_sentences_low_guarani_ratio":
                count_sentences_with_low_guarani_proportion(text),
            "num_sentences_with_javascript":
                count_sentences_with_javascript(text),
            "avg_words_in_capitalized_sentences":
                average_words_in_sentences_starting_with_capital(text),
            "num_bad_words_occurrences":
                count_bad_words_occurrences(text),
        }

        processed_records.append(record)

    # --------------------------------------------------------------
    # PRINT OLD METRICS ONCE (corpus-level)
    # --------------------------------------------------------------
    print("Old heuristic metrics (corpus-level):")
    for key, values in old_metrics.items():
        if values:
            print(f"  {key}: {sum(values) / len(values)}")

    write_jsonl(output_path, processed_records)

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

def main():
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        output_file = PROCESSED_DIR / jsonl_file.name
        process_file(jsonl_file, output_file)

    print("\n✔ Finished.")
    print("✔ Processed documents written to data/processed")

if __name__ == "__main__":
    main()
