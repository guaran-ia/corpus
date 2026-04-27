import os
import json
import sqlite3
import shutil
from datetime import datetime, timezone

from .utils.loader import load_jsonl
from .utils.writer import JSONLWriter
from .deduplication.url_dedup import normalize_url


class URLDeduplicator:
    def __init__(
        self,
        file_directory: str,
        output_base_dir: str,
        db_path: str = "url_seen.sqlite",
        exclude_files: list[str] | None = None,
        reuse_db: bool = True,
    ):
        self.file_directory = file_directory
        self.output_base_dir = output_base_dir
        self.db_path = db_path
        self.exclude_files = set(exclude_files or [])
        self.reuse_db = reuse_db

        self.conn = None
        self.cur = None

        self.run_dir = None
        self.dup_dir = None
        self.uniq_dir = None

        self.metrics = {
            "files": {},
            "global": {
                "total_docs": 0,
                "unique": 0,
                "duplicates": 0,
                "no_url": 0,
            },
        }

    # -------------------------
    # Run setup
    # -------------------------
    def _init_run_dirs(self):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.run_dir = os.path.join(self.output_base_dir, f"url_dedup_{ts}")
        self.dup_dir = os.path.join(self.run_dir, "duplicates")
        self.uniq_dir = os.path.join(self.run_dir, "non_duplicates")

        os.makedirs(self.dup_dir, exist_ok=True)
        os.makedirs(self.uniq_dir, exist_ok=True)

    # -------------------------
    # DB TODO: move this
    # -------------------------
    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS url_seen (
                url TEXT PRIMARY KEY
            )
        """)

        self.cur = self.conn.cursor()

    def close_db(self):
        self.conn.commit()
        self.conn.close()

        if not self.reuse_db:
            os.remove(self.db_path)

    # -------------------------
    # Public entry
    # -------------------------
    def run(self):
        if not os.path.isdir(self.file_directory):
            raise FileNotFoundError(self.file_directory)

        self._init_run_dirs()
        self.init_db()

        files = [f for f in os.listdir(self.file_directory) if f.endswith(".jsonl")]

        for file in files:
            self._process_file(file)

        self.close_db()
        self._write_report()

    # -------------------------
    # File processing
    # -------------------------
    def _process_file(self, file: str):
        file_id = os.path.splitext(file)[0]

        input_path = os.path.join(self.file_directory, file)
        dup_path = os.path.join(self.dup_dir, file)
        uniq_path = os.path.join(self.uniq_dir, file)

        self.metrics["files"][file_id] = {
            "total": 0,
            "unique": 0,
            "duplicates": 0,
            "no_url": 0,
        }

        if file_id in self.exclude_files:
            shutil.copyfile(input_path, uniq_path)
            return

        with JSONLWriter(dup_path) as w_dup, JSONLWriter(uniq_path) as w_uniq:
            for doc in load_jsonl(
                input_path,
                field_map={"text": "text", "id": "id"},
                load_metadata=True,
                metadata_fields="*",
            ):
                self._handle_doc(doc, file_id, w_dup, w_uniq)

    # -------------------------
    # Core logic
    # -------------------------
    def _handle_doc(self, doc, file_id, w_dup, w_uniq):
        self.metrics["global"]["total_docs"] += 1
        self.metrics["files"][file_id]["total"] += 1

        url = doc.metadata.get("url") if doc.metadata else None

        if not url or str(url).lower() in {"unknown", "nan", "none"}:
            self._write_unique(doc, w_uniq, file_id, no_url=True)
            return

        norm_url = normalize_url(url)

        if not norm_url:
            self._write_unique(doc, w_uniq, file_id, no_url=True)
            return

        try:
            self.cur.execute(
                "INSERT INTO url_seen(url) VALUES (?)",
                (norm_url,),
            )
            self._write_unique(doc, w_uniq, file_id)

        except sqlite3.IntegrityError:
            self._write_dup(doc, w_dup, file_id)

    # -------------------------
    # Writers + metrics
    # -------------------------
    def _write_unique(self, doc, writer, file_id, no_url=False):
        writer.write(doc, flatten=True, include_metadata=True, metadata_fields = "*")

        self.metrics["global"]["unique"] += 1
        self.metrics["files"][file_id]["unique"] += 1

        if no_url:
            self.metrics["global"]["no_url"] += 1
            self.metrics["files"][file_id]["no_url"] += 1

    def _write_dup(self, doc, writer, file_id):
        writer.write(doc, flatten=True)

        self.metrics["global"]["duplicates"] += 1
        self.metrics["files"][file_id]["duplicates"] += 1

    # -------------------------
    # Report
    # -------------------------
    def _write_report(self):
        report_path = os.path.join(self.run_dir, "report.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)