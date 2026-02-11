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
    average_sentence_length,
    average_ratio_of_symbols_to_words,
    average_ratio_of_stopwords_to_non_stopwords,
    average_character_repetition_ratio_per_sentence,
    average_word_repetition_ratio_per_sentence,
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

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_text(record: dict) -> str:
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def process_file(input_path: Path, output_path: Path):
    processed_records = []

    aggregated_metrics = {
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

        doc_metrics = {
            "avg_uppercase_letters_per_sentence":
                average_uppercase_letters_per_sentence(text),
            "avg_numbers_per_sentence":
                average_numbers_per_sentence(text),
            "avg_words_per_sentence":
                average_words_per_sentence(text),
            "avg_characters_per_sentence":
                average_characters_per_sentence(text),
            "avg_alphanumeric_characters_per_sentence":
                average_alphanumeric_characters_per_sentence(text),
            "mean_word_length":
                mean_word_length(text),
            "max_sentence_length":
                max_sentence_length(text),
            "min_sentence_length":
                min_sentence_length(text),
            "avg_sentence_length":
                average_sentence_length(text),
            "ratio_symbols_to_words":
                average_ratio_of_symbols_to_words(text),
            "ratio_stopwords_to_non_stopwords":
                average_ratio_of_stopwords_to_non_stopwords(text),
            "avg_character_repetition_ratio":
                average_character_repetition_ratio_per_sentence(text),
            "avg_word_repetition_ratio":
                average_word_repetition_ratio_per_sentence(text),

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

        aggregated_metrics["avg_uppercase_letters_per_sentence"].append(
            doc_metrics["avg_uppercase_letters_per_sentence"]
        )
        aggregated_metrics["avg_numbers_per_sentence"].append(
            doc_metrics["avg_numbers_per_sentence"]
        )
        aggregated_metrics["avg_words_per_sentence"].append(
            doc_metrics["avg_words_per_sentence"]
        )
        aggregated_metrics["avg_characters_per_sentence"].append(
            doc_metrics["avg_characters_per_sentence"]
        )
        aggregated_metrics["avg_alphanumeric_characters_per_sentence"].append(
            doc_metrics["avg_alphanumeric_characters_per_sentence"]
        )
        aggregated_metrics["mean_word_length"].append(
            doc_metrics["mean_word_length"]
        )
        aggregated_metrics["max_sentence_length"].append(
            doc_metrics["max_sentence_length"]
        )
        aggregated_metrics["min_sentence_length"].append(
            doc_metrics["min_sentence_length"]
        )
        aggregated_metrics["avg_sentence_length"].append(
            doc_metrics["avg_sentence_length"]
        )
        aggregated_metrics["ratio_symbols_to_words"].append(
            doc_metrics["ratio_symbols_to_words"]
        )
        aggregated_metrics["ratio_stopwords_to_non_stopwords"].append(
            doc_metrics["ratio_stopwords_to_non_stopwords"]
        )
        aggregated_metrics["avg_character_repetition_ratio"].append(
            doc_metrics["avg_character_repetition_ratio"]
        )
        aggregated_metrics["avg_word_repetition_ratio"].append(
            doc_metrics["avg_word_repetition_ratio"]
        )

        record["heuristics"] = {
            "num_lorem_ipsum_sentences":
                doc_metrics["num_lorem_ipsum_sentences"],
            "num_sentences_ending_with_ellipsis":
                doc_metrics["num_sentences_ending_with_ellipsis"],
            "num_sentences_starting_with_bullet":
                doc_metrics["num_sentences_starting_with_bullet"],
            "num_sentences_without_terminal_punctuation":
                doc_metrics["num_sentences_without_terminal_punctuation"],
            "num_sentences_with_curly_bracket":
                doc_metrics["num_sentences_with_curly_bracket"],
            "num_sentences_with_legal_phrases":
                doc_metrics["num_sentences_with_legal_phrases"],
            "num_sentences_low_guarani_ratio":
                doc_metrics["num_sentences_low_guarani_ratio"],
            "num_sentences_with_javascript":
                doc_metrics["num_sentences_with_javascript"],
            "avg_words_in_capitalized_sentences":
                doc_metrics["avg_words_in_capitalized_sentences"],
            "num_bad_words_occurrences":
                doc_metrics["num_bad_words_occurrences"],
        }

        processed_records.append(record)

    print("Heuristic metrics:")
    for key, values in aggregated_metrics.items():
        if values:
            print(f"  {key}: {sum(values) / len(values)}")

    write_jsonl(output_path, processed_records)


def main():
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        output_file = PROCESSED_DIR / jsonl_file.name
        process_file(jsonl_file, output_file)

    print("\n✔ Finished.")
    print("✔ Processed documents written to data/processed")


if __name__ == "__main__":
    main()
