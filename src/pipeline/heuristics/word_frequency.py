import json
from collections import Counter
from pathlib import Path

from .tokenization import tokenize

DATA_DIR = Path("data/raw")


def extract_text(record: dict) -> str:
    """
    Extrae texto de campos comunes en los corpora GuaranIA.
    """
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def word_frequencies() -> Counter:
    """
    Calcula frecuencias de palabras sobre todos los corpora en data/raw.
    (Exploratorio: NO filtra stopwords)
    """
    counter = Counter()

    for path in DATA_DIR.glob("*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = extract_text(record)
                if not text:
                    continue

                for token in tokenize(text):
                    counter[token] += 1

    return counter


if __name__ == "__main__":
    freq = word_frequencies()
    for word, count in freq.most_common(200):
        print(word, count)
