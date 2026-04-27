import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlunparse


def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize a URL string to enable consistent duplicate detection.

    This function standardizes URLs before comparison. It removes
    leading/trailing spaces, ignores empty or placeholder values,
    adds an HTTP scheme to URLs starting with 'www.', lowercases
    the scheme and domain, removes trailing slashes from the path,
    and preserves the query string.

    Args:
        url (Optional[str]): The original URL string from a corpus record.

    Returns:
        Optional[str]: The normalized URL if valid, otherwise None.
    """
    if not url:
        return None

    url = url.strip()

    if not url or url.lower() in {"unknown", "nan", "none"}:
        return None

    if url.startswith("www."):
        url = "http://" + url

    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = parsed.query

    return urlunparse((scheme, netloc, path, "", query, ""))


def execute_url_deduplication(data_dir: str, output_dir: str) -> None:
    """
    Perform URL-based deduplication across all processed corpora.

    This function reads all JSONL files in the processed directory,
    normalizes the URL field of each document, and detects duplicate
    groups globally across all corpora.

    It generates three output files:
        - duplicate_urls.json: mapping from normalized URL to document IDs
        - duplicate_ids.json: mapping from document ID to duplicate document IDs
        - report.json: summary statistics of the deduplication run

    It also updates each processed document with URL duplicate metadata
    under the following structure:

        "duplicate": {
            "url": {
                "has_duplicate": true/false,
                "docs_ids": [...]
            }
        }

    Args:
        data_dir (str): Base directory containing the processed corpora.
        output_dir (str): Directory where deduplication outputs will be saved.

    Returns:
        None
    """
    start_time = datetime.now()
    start_timestamp = time.time()

    processed_dir = os.path.join(data_dir, "processed")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    dedup_dir = os.path.join(output_dir, "deduplication", f"url_{timestamp}")
    os.makedirs(dedup_dir, exist_ok=True)

    duplicate_urls_path = os.path.join(dedup_dir, "duplicate_urls.json")
    duplicate_ids_path = os.path.join(dedup_dir, "duplicate_ids.json")
    report_path = os.path.join(dedup_dir, "report.json")

    url_map: Dict[str, List[str]] = {}
    total_docs = 0

    corpus_files = sorted(
        f for f in os.listdir(processed_dir)
        if f.endswith(".jsonl")
    )

    # Build a global mapping from normalized URL to document IDs.
    for corpus_file in corpus_files:
        corpus_path = os.path.join(processed_dir, corpus_file)
        corpus_name = os.path.splitext(corpus_file)[0]

        with open(corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                record = json.loads(line)
                total_docs += 1

                doc_id = record.get("id")
                if not doc_id:
                    doc_id = f"{corpus_name}_{idx}"

                normalized = normalize_url(record.get("url"))

                if normalized:
                    url_map.setdefault(normalized, []).append(doc_id)

    duplicate_urls: Dict[str, List[str]] = {}
    duplicate_ids: Dict[str, List[str]] = {}

    duplicate_groups = 0
    total_duplicate_documents = 0

    # Detect duplicated URL groups and derive the document-to-document mapping.
    for url, ids in url_map.items():
        if len(ids) > 1:
            duplicate_groups += 1
            total_duplicate_documents += len(ids)

            duplicate_urls[url] = ids

            for doc_id in ids:
                duplicate_ids[doc_id] = [
                    other_id for other_id in ids if other_id != doc_id
                ]

    with open(duplicate_urls_path, "w", encoding="utf-8") as f:
        json.dump(duplicate_urls, f, ensure_ascii=False, indent=2)

    with open(duplicate_ids_path, "w", encoding="utf-8") as f:
        json.dump(duplicate_ids, f, ensure_ascii=False, indent=2)

    # Update each processed corpus file with duplicate metadata.
    for corpus_file in corpus_files:
        corpus_path = os.path.join(processed_dir, corpus_file)
        corpus_name = os.path.splitext(corpus_file)[0]
        updated_records = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                record = json.loads(line)

                doc_id = record.get("id")
                if not doc_id:
                    doc_id = f"{corpus_name}_{idx}"
                    record["id"] = doc_id

                record["duplicate"] = {
                    "url": {
                        "has_duplicate": doc_id in duplicate_ids,
                        "docs_ids": duplicate_ids.get(doc_id, []),
                    }
                }

                updated_records.append(record)

        with open(corpus_path, "w", encoding="utf-8") as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Updated: {corpus_file}")

    end_time = datetime.now()
    duration_minutes = (time.time() - start_timestamp) / 60

    report = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_minutes": round(duration_minutes, 2),
        "method": "url",
        "total_docs": total_docs,
        "total_unique_urls": len(url_map),
        "duplicate_groups": duplicate_groups,
        "total_duplicate_documents": total_duplicate_documents,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nURL deduplication completed.")