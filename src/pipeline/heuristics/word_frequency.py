import json
from collections import Counter
from pathlib import Path

from nltk.tokenize import TweetTokenizer

from .stopwords import STOPWORDS

DATA_DIR = Path("data/raw")

# Tokenizador oficial del pipeline
_tokenizer = TweetTokenizer(preserve_case=False)


def tokenize(text: str):
    """
    Tokeniza texto usando TweetTokenizer.
    Devuelve solo tokens alfabéticos (unicode).
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token


def extract_text(record: dict) -> str:
    """
    Extrae texto de campos comunes en los corpora GuaranIA.
    """
    for key in ("text", "sentence", "content"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def word_frequencies(exclude_stopwords: bool = False) -> Counter:
    """
    Calcula frecuencias de palabras sobre todos los corpora en data/raw.

    Args:
        exclude_stopwords: si True, excluye STOPWORDS del conteo
    """
    counter = Counter()

    for path in DATA_DIR.glob("*.jsonl"):
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                text = extract_text(record)
                if not text:
                    continue

                for token in tokenize(text):
                    if exclude_stopwords and token in STOPWORDS:
                        continue
                    counter[token] += 1

    return counter


# ------------------------------------------------------------------
# EXPLORATION: print most frequent words
# ------------------------------------------------------------------
if __name__ == "__main__":
    freq = word_frequencies(exclude_stopwords=True)
    for word, count in freq.most_common(200):
        print(word, count)
