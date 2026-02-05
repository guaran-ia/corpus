import json
from pathlib import Path
from typing import Dict
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
    average_sentences_per_document,
    min_sentences_per_document,
    max_sentences_per_document,
)

# ---------------------------------------------------------------------
# DATA DIRECTORY
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("data/reports/document_metrics_report.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True) 

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

def compute_metrics_for_text(text: str) -> Dict[str, float]:
    """Compute all the requested metrics for a text."""
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
    }
def run_on_file(path: Path) -> dict:
    """
    Run all heuristics on a single corpus file.
    """
    texts = []
    output_path = PROCESSED_DIR / path.name

    with path.open("r", encoding="utf-8") as infile, \
        output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue

            record = json.loads(line)
            text = extract_text(record)
            if text:
                metrics = compute_metrics_for_text(text)
                record.update(metrics)
                texts.append(text)
            else:
                pass
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    document_metrics = {
        "avg_sentences_per_document": average_sentences_per_document(texts),
        "min_sentences_per_document": min_sentences_per_document(texts),
        "max_sentences_per_document": max_sentences_per_document(texts),
    }

    return {
        "corpus": path.stem,
        "document_metrics": document_metrics,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    reports = []
    for jsonl_file in DATA_DIR.glob("*.jsonl"):
        print(f"\nProcessing corpus: {jsonl_file.name}")
        results = run_on_file(jsonl_file)

        if not results:
            print("No valid texts found.")
            continue
        reports.append(results)

        print("Document-level metrics:")
        for k, v in results["document_metrics"].items():
            print(f"  {k}: {v}")
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print(f"\nDocument metrics report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
