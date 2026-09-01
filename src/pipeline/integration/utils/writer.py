import json
import os

class JSONLWriter:
    def __init__(self, filepath, mode="w", encoding="utf-8"):
        self.filepath = filepath
        self.mode = mode
        self.encoding = encoding
        self._file = None
        self._has_written = False

    def __enter__(self):
        return self

    def _ensure_open(self):
        if self._file is None:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self._file = open(self.filepath, self.mode, encoding=self.encoding)

    def write(self, obj, **doc_kwargs):
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict(**doc_kwargs)

        if not self._has_written:
            self._ensure_open()

        line = json.dumps(obj, ensure_ascii=False)
        self._file.write(line + "\n")
        self._has_written = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None