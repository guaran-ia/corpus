import sqlite3
import pickle
import os
from datasketch import MinHash

class MinHashSQLiteStore:
    def __init__(self, db_path: str, num_perm: int, reuse: bool = True):
        self.db_path = db_path
        self.num_perm = num_perm
        self.reuse = reuse

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS minhashes (
                id TEXT PRIMARY KEY,
                hash BLOB
            )
        """)

    def exists(self, key: str) -> bool:
        self.cur.execute("SELECT 1 FROM minhashes WHERE id = ?", (key,))
        return self.cur.fetchone() is not None

    def load(self, key: str) -> MinHash | None:
        self.cur.execute("SELECT hash FROM minhashes WHERE id = ?", (key,))
        row = self.cur.fetchone()
        if not row:
            return None

        return pickle.loads(row[0])

    def save(self, key: str, minhash: MinHash):
        blob = pickle.dumps(minhash)
        self.cur.execute(
            "INSERT OR REPLACE INTO minhashes (id, hash) VALUES (?, ?)",
            (key, blob),
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

        if not self.reuse:
            os.remove(self.db_path)