from typing import Any, Dict, Optional, Iterable, Iterator
from .document import Document
from .hashing import make_doc_id
import json
import csv


def _process_row(
    row: Dict[str, Any],
    field_map: Dict[str, str],
    load_metadata: bool = False,
    metadata_fields: Optional[Iterable[str] | str] = None,
    rename_map: Optional[Dict[str, str]] = None,
    generate_id: bool = True,
) -> Document:

    rename_map = rename_map or {}

    if "text" not in field_map:
        raise ValueError("field_map must contain a 'text' key")

    # Validate and extract mapped fields

    data = {}

    for attr, source_key in field_map.items():
        if source_key is not None and source_key not in row:
            raise ValueError(f"Column '{source_key}' (mapped to '{attr}') not found in row")
        data[attr] = row.get(source_key)


    # ----------------------------
    # 2. Validate required fields
    # ----------------------------
    text = data.get("text")
    if not text:
        raise ValueError("Missing required field: text")

    doc_id = data.get("id", None)

    if not doc_id:
        if generate_id:
            doc_id = make_doc_id(text)
        else:
            raise ValueError("Missing required field: id (and generate_id=False)")

    # ----------------------------
    # 3. Metadata extraction
    # ----------------------------
    metadata = {}

    if load_metadata:
        if metadata_fields == "*":
            metadata_fields_set = set(row.keys()) - set(field_map.values())

        elif isinstance(metadata_fields, str):
            metadata_fields_set = {metadata_fields}

        elif metadata_fields is None:
            metadata_fields_set = set(row.keys()) - set(field_map.values())

        else:
            metadata_fields_set = set(metadata_fields)

        for k in metadata_fields_set:
            if k not in row:
                raise ValueError(f"Metadata field '{k}' not found in row")

            new_key = rename_map.get(k, k)
            metadata[new_key] = row[k]

    # ----------------------------
    # 4. Build Document
    # ----------------------------
    return Document(
        id=doc_id,
        text=text,
        metadata=metadata or None,
    )


def load_jsonl(
    file_path: str,
    field_map: dict,
    load_metadata: bool = False,
    metadata_fields: list[str] | str | None = None,
    rename_map: dict[str, str] | None = None,
    generate_id: bool = True,
) -> Iterator[Document]:

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                row = json.loads(line)

                yield _process_row(
                    row,
                    field_map,
                    load_metadata=load_metadata,
                    metadata_fields=metadata_fields,
                    rename_map=rename_map,
                    generate_id=generate_id,
                )

            except Exception as e:
                raise ValueError(
                    f"Error on JSONL line {line_number}: {e}"
                ) from e


def load_csv(
    file_path: str,
    field_map: dict,
    load_metadata: bool = False,
    metadata_fields: list[str] | str | None = None,
    rename_map: dict[str, str] | None = None,
    generate_id: bool = True,
) -> Iterator[Document]:

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for line_number, row in enumerate(reader, start=2):
            try:
                yield _process_row(
                    row,
                    field_map,
                    load_metadata=load_metadata,
                    metadata_fields=metadata_fields,
                    rename_map=rename_map,
                    generate_id=generate_id,
                )

            except Exception as e:
                raise ValueError(
                    f"Error on CSV row {line_number}: {e}"
                ) from e