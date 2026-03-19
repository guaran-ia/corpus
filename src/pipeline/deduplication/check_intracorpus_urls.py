import os
import json
from collections import defaultdict

DATA_DIR = "data/processed"
OUTPUT_FILE = "outputs/deduplication/url_dedup_ignore_corpora.json"


def analyze_corpus(corpus_path):
    """
    Analyze a single corpus and detect URLs that appear multiple times
    with different texts (intracorpus comparison).
    """

    url_map = defaultdict(list)

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            url = record.get("url")
            text = record.get("text")

            if url and text and url.lower() not in {"unknown", "nan", "none"}:
                url_map[url].append(text)

    problematic_urls = 0

    for url, texts in url_map.items():
        if len(texts) > 1:
            unique_texts = set(texts)

            if len(unique_texts) > 1:
                problematic_urls += 1

    return problematic_urls


def main():
    corpus_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".jsonl")
    ]

    ignore_corpora = []

    for corpus_file in corpus_files:
        corpus_path = os.path.join(DATA_DIR, corpus_file)

        count = analyze_corpus(corpus_path)

        if count > 0:
            corpus_name = corpus_file.replace(".jsonl", "")
            ignore_corpora.append(corpus_name)

            print(f"{corpus_name}: {count} URLs")

    os.makedirs("outputs/deduplication", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ignore_corpora, f, indent=2)

    print("\nIgnore corpus list saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
