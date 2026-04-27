import os
import json
from collections import defaultdict
from datetime import datetime, timezone

from datasketch import MinHash, MinHashLSH

# your existing utilities
from .utils.loader import load_jsonl
from .utils.writer import JSONLWriter
from .deduplication.utils import canonicalize_text
from utils.mh_utils import get_shingles


class MinHashDeduplicator:
    def __init__(
        self,
        input_dir: str,
        output_base_dir: str,
        minhash_db_path: str = "minhash.sqlite",
        shingle_size: int = 5,
        similarity_threshold: float = 0.8,
        num_perm: int = 128,
        reuse_minhash: bool = True,
    ):
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir

        self.shingle_size = shingle_size
        self.threshold = similarity_threshold
        self.num_perm = num_perm
        self.reuse_minhash = reuse_minhash

        #global LSH (cross-file dedup)
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # file writers cache
        self.writers = {}

        # metrics
        self.metrics = {
            "total_docs": 0,
            "unique_docs": 0,
            "duplicate_docs": 0,
        }

    def _init_run_dirs(self):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.run_dir = os.path.join(self.output_base_dir, f"minhash_dedup_{ts}")
        self.dup_dir = os.path.join(self.run_dir, "duplicates")
        self.uniq_dir = os.path.join(self.run_dir, "non_duplicates")
        self.shingle_dir = os.path.join(self.run_dir, "shingles")

        os.makedirs(self.dup_dir, exist_ok=True)
        os.makedirs(self.uniq_dir, exist_ok=True)

        if self.store_shingles:
            os.makedirs(self.shingle_dir, exist_ok=True)
    
    def run(self):
        self._init_run_dirs()

        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]

        for file in files:
            self._process_file(file)

        self._write_report()

    def _compute_minhash(self, shingles):
        m = MinHash(num_perm=self.num_perm)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        return m

    def stage1_compute(self, doc, shingle_size, num_perm):
        global_id = f"{doc.id}"

        shingles = get_shingles(doc.text, shingle_size)
        minhash = self._compute_minhash(shingles, num_perm)

        return global_id, doc, minhash

    def _process_file(self, file: str):
        input_path = os.path.join(self.input_dir, file)

        dup_path = os.path.join(self.dup_dir, file)
        uniq_path = os.path.join(self.uniq_dir, file)

        w_dup = JSONLWriter(dup_path)
        w_uniq = JSONLWriter(uniq_path)

        self.writers[file] = (w_dup, w_uniq)

        with w_dup, w_uniq:
            for doc in load_jsonl(input_path):
                self._process_doc(doc, file, w_dup, w_uniq)
    
    
    def _process_doc(self, doc, file, w_dup, w_uniq):
        self.metrics["total_docs"] += 1

        doc_id = doc["id"]
        text = doc["text"]

        shingles = self._get_shingles(text)

        if self.store_shingles:
            self._save_shingles(doc_id, file, shingles)

        minhash = self._compute_minhash(shingles)

        # IMPORTANT: query BEFORE insert
        candidates = self.lsh.query(minhash)

        is_duplicate = False

        for cand_id in candidates:
            # we don't store shingles globally, so assume they are embedded or recomputed elsewhere
            # simplest approach: store lightweight index in memory
            cand_shingles = self._get_cached_shingles(cand_id)

            if cand_shingles:
                sim = self._jaccard(shingles, cand_shingles)

                if sim >= self.threshold:
                    is_duplicate = True
                    break

        # write result
        if is_duplicate:
            w_dup.write(doc, flatten=True)
            self.metrics["duplicate_docs"] += 1
        else:
            w_uniq.write(doc, flatten=True)
            self.metrics["unique_docs"] += 1

        # insert AFTER query
        self.lsh.insert(doc_id, minhash)
    
    def _get_shingles(self, text: str):
        text = canonicalize_text(text)
        tokens = text.split()

        return {
            " ".join(tokens[i:i+self.shingle_size])
            for i in range(len(tokens) - self.shingle_size + 1)
        }

    def _compute_minhash(self, shingles):
        m = MinHash(num_perm=self.num_perm)

        for sh in shingles:
            m.update(sh.encode("utf-8"))

        return m
    
    def _jaccard(self, a, b):
        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    def _save_shingles(self, doc_id, file, shingles):
        path = os.path.join(self.shingle_dir, file)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": doc_id,
                "shingles": list(shingles)
            }, f)

    def _write_report(self):
        report = {
            "total_docs": self.metrics["total_docs"],
            "unique_docs": self.metrics["unique_docs"],
            "duplicate_docs": self.metrics["duplicate_docs"],
            "params": {
                "shingle_size": self.shingle_size,
                "threshold": self.threshold,
                "num_perm": self.num_perm,
            }
        }

        with open(os.path.join(self.run_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)