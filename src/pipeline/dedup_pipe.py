from utils.loader import load_jsonl
import os
from deduplication.url_dedup import normalize_url
import sqlite3
import shutil
from utils.writer import flatten_document
import json

url_dedup_ignore = [
  "hplt-3",
  "orembae",
  "FinePDF",
  "fineweb-2",
  "moscar",
  "flores-200",
  "multi-wiki-qa"
]

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS url_seen (
            url TEXT PRIMARY KEY
        )
    """)
    return conn

def url_deduplication(
    file_directory: str,
    exclude_files: list[str],
    output_directory_duplicates: str,
    output_directory_non_duplicates: str,
    db_path: str = "url_seen.sqlite"
):

    if not os.path.isdir(file_directory):
        print(f"Directory not found: {file_directory}")
        return

    os.makedirs(output_directory_duplicates, exist_ok=True)
    os.makedirs(output_directory_non_duplicates, exist_ok=True)

    conn = init_db(db_path)
    cur = conn.cursor()

    files = [
        f for f in os.listdir(file_directory)
        if f.endswith(".jsonl")
    ]

    for file in files:

        file_id = os.path.splitext(file)[0]

        input_path = os.path.join(file_directory, file)
        dup_out_path = os.path.join(output_directory_duplicates, file)
        uniq_out_path = os.path.join(output_directory_non_duplicates, file)

        # -------------------------
        # Skip excluded files (copy as-is)
        # -------------------------
        if file_id in exclude_files:
            shutil.copyfile(input_path, uniq_out_path)
            continue

        # -------------------------
        # Stream documents (IMPORTANT CHANGE)
        # -------------------------
        with open(dup_out_path, "w", encoding="utf-8") as f_dup, \
             open(uniq_out_path, "w", encoding="utf-8") as f_unique:

            for doc in load_jsonl(input_path, field_map={"text":"text", "id":"id"}, load_metadata=True, metadata_fields="*"):

                # -------------------------
                # extract URL from metadata
                # -------------------------
                url = None
                if doc.metadata:
                    url = doc.metadata.get("url")

                if not url or str(url).lower() in {"unknown", "nan", "none"}:
                    # no URL → treat as unique
                    f_unique.write(json.dumps(flatten_document(doc), ensure_ascii=False) + "\n")
                    continue

                norm_url = normalize_url(url)

                if not norm_url:
                    f_unique.write(json.dumps(flatten_document(doc), ensure_ascii=False) + "\n")
                    continue

                # -------------------------
                # global dedup check (disk-backed)
                # -------------------------
                try:
                    cur.execute(
                        "INSERT INTO url_seen(url) VALUES (?)",
                        (norm_url,)
                    )
                    f_unique.write(flatten_document(doc) + "\n")

                except sqlite3.IntegrityError:
                    f_dup.write(flatten_document(doc) + "\n")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    data_path = "data/processed"
    output_duplicates = "outputs/url_deduplication/duplicates"
    output_non_duplicates = "outputs/url_deduplication/non_duplicates"

    url_deduplication(
        data_path, url_dedup_ignore, output_duplicates, output_non_duplicates
    )