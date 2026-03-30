import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

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
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "heuristics_report.md"

HEURISTIC_KEYS = [
    "avg_uppercase_letters_per_sentence",
    "avg_numbers_per_sentence",
    "avg_words_per_sentence",
    "avg_characters_per_sentence",
    "avg_alphanumeric_characters_per_sentence",
    "mean_word_length",
    "avg_sentence_length",
    "max_sentence_length",
    "min_sentence_length",
    "ratio_symbols_to_words",
    "ratio_stopwords_to_non_stopwords",
    "avg_character_repetition_ratio_per_sentence",
    "avg_word_repetition_ratio_per_sentence",
    "count_lorem_ipsum_sentences",
    "count_sentences_ending_with_ellipsis",
    "count_sentences_starting_with_bullet",
    "count_sentences_without_terminal_punctuation",
    "count_sentences_with_curly_bracket",
    "count_sentences_with_legal_phrases",
    "count_sentences_with_low_guarani_proportion",
    "count_sentences_with_javascript",
    "average_words_in_sentences_starting_with_capital",
    "count_bad_words_occurrences",
]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_text(record: dict) -> str:
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def compute_metrics(text: str) -> Dict[str, float]:
    return {
        "avg_uppercase_letters_per_sentence": average_uppercase_letters_per_sentence(text),
        "avg_numbers_per_sentence": average_numbers_per_sentence(text),
        "avg_words_per_sentence": average_words_per_sentence(text),
        "avg_characters_per_sentence": average_characters_per_sentence(text),
        "avg_alphanumeric_characters_per_sentence": average_alphanumeric_characters_per_sentence(text),
        "mean_word_length": mean_word_length(text),
        "avg_sentence_length": average_sentence_length(text),
        "max_sentence_length": max_sentence_length(text),
        "min_sentence_length": min_sentence_length(text),
        "ratio_symbols_to_words": average_ratio_of_symbols_to_words(text),
        "ratio_stopwords_to_non_stopwords": average_ratio_of_stopwords_to_non_stopwords(text),
        "avg_character_repetition_ratio_per_sentence": average_character_repetition_ratio_per_sentence(text),
        "avg_word_repetition_ratio_per_sentence": average_word_repetition_ratio_per_sentence(text),
        "count_lorem_ipsum_sentences": count_lorem_ipsum_sentences(text),
        "count_sentences_ending_with_ellipsis": count_sentences_ending_with_ellipsis(text),
        "count_sentences_starting_with_bullet": count_sentences_starting_with_bullet(text),
        "count_sentences_without_terminal_punctuation": count_sentences_without_terminal_punctuation(text),
        "count_sentences_with_curly_bracket": count_sentences_with_curly_bracket(text),
        "count_sentences_with_legal_phrases": count_sentences_with_legal_phrases(text),
        "count_sentences_with_low_guarani_proportion": count_sentences_with_low_guarani_proportion(text),
        "count_sentences_with_javascript": count_sentences_with_javascript(text),
        "average_words_in_sentences_starting_with_capital": average_words_in_sentences_starting_with_capital(text),
        "count_bad_words_occurrences": count_bad_words_occurrences(text),
    }


def compute_corpus_results(path: Path) -> Optional[Dict]:
    values_by_metric: Dict[str, List[float]] = {key: [] for key in HEURISTIC_KEYS}
    documents_count = 0

    for record in read_jsonl(path):
        text = extract_text(record)
        if not text:
            continue

        metrics = compute_metrics(text)
        documents_count += 1

        for key, value in metrics.items():
            values_by_metric[key].append(value)

    if documents_count == 0:
        return None

    corpus_averages = {
        key: mean(values)
        for key, values in values_by_metric.items()
        if values
    }

    return {
        "corpus": path.stem,
        "documents_count": documents_count,
        "metrics": corpus_averages,
    }


def compute_overall_results(corpus_results: List[Dict]) -> Dict[str, float]:
    overall_values: Dict[str, List[float]] = {key: [] for key in HEURISTIC_KEYS}

    for corpus_result in corpus_results:
        for key, value in corpus_result["metrics"].items():
            overall_values[key].append(value)

    return {
        key: mean(values)
        for key, values in overall_values.items()
        if values
    }


def build_markdown_report(
    corpus_results: List[Dict],
    overall_results: Dict[str, float],
) -> str:
    lines: List[str] = []

    lines.append("# Heuristics Report")
    lines.append("")

    header = ["Corpus"] + HEURISTIC_KEYS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * len(header))

    for corpus_result in sorted(corpus_results, key=lambda x: x["corpus"].lower()):
        row = [corpus_result["corpus"]]

        for key in HEURISTIC_KEYS:
            value = corpus_result["metrics"].get(key)
            if value is not None:
                row.append(f"{value:.4f}")
            else:
                row.append("")

        lines.append("| " + " | ".join(row) + " |")

    overall_row = ["Overall"]

    for key in HEURISTIC_KEYS:
        value = overall_results.get(key)
        if value is not None:
            overall_row.append(f"{value:.4f}")
        else:
            overall_row.append("")

    lines.append("| " + " | ".join(overall_row) + " |")

    return "\n".join(lines)


def write_report(extra_raw_files: Optional[List[Path]] = None) -> Path:
    files = sorted(RAW_DIR.glob("*.jsonl"))

    if extra_raw_files:
        files.extend(extra_raw_files)

    corpus_results: List[Dict] = []

    for path in files:
        result = compute_corpus_results(path)
        if result is not None:
            corpus_results.append(result)

    overall_results = compute_overall_results(corpus_results)
    report = build_markdown_report(corpus_results, overall_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    return REPORT_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coreguapa",
        type=Path,
        help="External path to the coreguapa JSONL corpus.",
    )
    args = parser.parse_args()

    extra_files: List[Path] = []

    if args.coreguapa:
        if not args.coreguapa.exists():
            raise FileNotFoundError(f"Coreguapa file not found: {args.coreguapa}")
        extra_files.append(args.coreguapa)

    report_path = write_report(extra_raw_files=extra_files)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
