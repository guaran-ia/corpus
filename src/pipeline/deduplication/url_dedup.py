import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """
    Normalize a URL string to enable consistent duplicate detection.

    This function cleans and standardizes URLs before comparison. It removes
    leading/trailing spaces, ensures the URL has a valid scheme and domain,
    converts scheme and domain to lowercase, and removes trailing slashes
    from the path.

    Args:
        url (str): Original URL string from the corpus record.

    Returns:
        str: Normalized URL.

    Raises:
        ValueError: If the URL is invalid or missing required components.
    """

    url = url.strip()

    if not url or url.lower() in {"unknown", "nan", "none"}:
        raise ValueError("Invalid URL value")

    if url.startswith("www."):
        url = "http://" + url

    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format: {url}")

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = parsed.query

    return urlunparse((scheme, netloc, path, "", query, ""))


def execute_url_deduplication(data_dir: str, output_dir: str) -> None:
    """
    Identify URL-based deduplication across all processed corpora.

    This function reads all JSONL corpus files from the processed data
    directory, extracts document IDs and URLs, normalizes the URLs,
    and groups documents that share the same URL.

    Duplicate information is saved into output files along with a report
    summarizing the deduplication process.

    Args:
        data_dir (str): Base directory containing the processed corpora.
        output_dir (str): Directory where deduplication outputs will be written.

    Outputs:
        duplicate_ids.json:
            Mapping from document ID to other document IDs sharing the same URL.

        duplicate_urls.json:
            Mapping from normalized URL to all document IDs sharing that URL.

        report.json:
            Summary statistics including number of documents processed,
            unique URLs, duplicate groups, and runtime.

    Notes:
        - Only documents with valid URLs are considered.
        - URLs are normalized before comparison.
        - A new timestamped output directory is created for each run.
    """

    start_time = datetime.now()
    start_timestamp = time.time()

    processed_dir = os.path.join(data_dir, "processed")

    # timestamp format aligned with minhash
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    dedup_dir = os.path.join(output_dir, "deduplication", f"url_{timestamp}")
    os.makedirs(dedup_dir, exist_ok=True)

    url_map: Dict[str, List[str]] = {}
    total_docs = 0

    corpus_files = [
        f for f in os.listdir(processed_dir)
        if f.endswith(".jsonl")
    ]

    for corpus_file in corpus_files:
        corpus_path = os.path.join(processed_dir, corpus_file)

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                total_docs += 1

                normalized = normalize_url(record.get("url"))
                doc_id = record.get("id")

                if normalized and doc_id:
                    url_map.setdefault(normalized, []).append(doc_id)

    duplicate_ids: Dict[str, List[str]] = {}
    duplicate_urls: Dict[str, List[str]] = {}

    duplicate_groups = 0
    total_duplicate_documents = 0
    # Iterate over normalized URLs and their associated document IDs
    # to identify URLs that appear in more than one document (duplicates).

    for url, ids in url_map.items():
        if len(ids) > 1:
            duplicate_groups += 1
            total_duplicate_documents += len(ids)

            duplicate_urls[url] = ids

            for doc_id in ids:
                duplicate_ids[doc_id] = [
                    other_id for other_id in ids if other_id != doc_id
                ]

    duplicate_ids_path = os.path.join(dedup_dir, "duplicate_ids.json")

    with open(duplicate_ids_path, "w", encoding="utf-8") as f:
        json.dump(duplicate_ids, f, ensure_ascii=False, indent=2)

    duplicate_urls_path = os.path.join(dedup_dir, "duplicate_urls.json")

    with open(duplicate_urls_path, "w", encoding="utf-8") as f:
        json.dump(duplicate_urls, f, ensure_ascii=False, indent=2)

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

    report_path = os.path.join(dedup_dir, "report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("URL deduplication completed.")
