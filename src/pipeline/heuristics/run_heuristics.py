import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

OUTPUT_DIR = Path("data/processed/processed_output.jsonl")
# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def read_jsonl(path: Path):
    """Yield JSON records from a .jsonl file."""
    with path.open() as f:
        for line in f:
            if not line:
                continue
            yield json.loads(line)

def append_jsonl(path: Path, records: Iterable[Dict]) -> None:
    """Add records to a .jsonl file (creates the file if it does not exist)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
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

def augment_record(record: Dict) -> Dict:
    """Extend the JSON record with metrics if there is text."""
    text = extract_text(record)
    if not text:
        return record
    metrics = compute_metrics_for_text(text)
    return {**record, **metrics}


def process_all(input_root: Path, out_file: Path) -> Tuple[int, int]:
    """
    Process all .jsonl files under the given root and write everything processed to a single JSONL file.

    Returns (processed) where:
    - processed: total number of records read across all input .jsonl files    """
    processed = 0

    # Rewrite fresh output on each run
    if out_file.exists():
        out_file.unlink()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for in_path in input_root.rglob("*.jsonl"):
        batch: List[Dict] = []
        for rec in read_jsonl(in_path):
            processed += 1
            new_rec = augment_record(rec)
            # If text could be extracted, new_rec will include metrics
            batch.append(new_rec)

            # Batch flushing to avoid excessive memory usage
            if len(batch) >= 2000:
                append_jsonl(out_file, batch)
                batch.clear()

        if batch:
            append_jsonl(out_file, batch)

    return processed
# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Processing: {DATA_DIR}")
    print(f"Writing consolidated output to: {OUTPUT_DIR}")
    processed = process_all(DATA_DIR, OUTPUT_DIR)
    print("\nSummary:")
    print(f"Total processed: {processed}")

if __name__ == "__main__":
    main()