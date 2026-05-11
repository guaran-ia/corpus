from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable


@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(
        self,
        include_metadata: bool = True,
        metadata_fields: Optional[Iterable[str] | str] = None,
        rename_map: Optional[Dict[str, str]] = None,
        flatten: bool = True,
    ) -> Dict[str, Any]:

        data = {
            "id": self.id,
            "text": self.text,
        }

        if include_metadata and self.metadata:
            metadata = dict(self.metadata)

            if isinstance(metadata_fields, str):
                metadata_fields = metadata_fields.strip()

                if metadata_fields == "*":
                    metadata_fields = list(metadata.keys())
                else:
                    metadata_fields = [metadata_fields]

            if metadata_fields is not None:
                metadata_fields_set = set(metadata_fields)
                metadata = {
                    k: v for k, v in metadata.items()
                    if k in metadata_fields_set
                }

            # -----------------------------
            # Flatten or nest
            # -----------------------------
            if flatten:
                for k, v in metadata.items():
                    if k not in data:
                        data[k] = v
            else:
                data["metadata"] = metadata

        return self._apply_rename(data, rename_map)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        field_map: Dict[str, str],
        load_metadata: bool = False,
        metadata_fields: Optional[Iterable[str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
    ):
        """
        Reconstruct Document from dict with symmetric logic.
        """

        # Reverse rename_map if provided
        if rename_map:
            reverse_map = {v: k for k, v in rename_map.items()}
            data = {reverse_map.get(k, k): v for k, v in data.items()}

        # Extract required fields
        id_ = data[field_map["id"]]
        text_ = data[field_map["text"]]

        metadata = None

        if load_metadata:
            metadata = {}

            known_fields = set(field_map.values())

            if isinstance(metadata_fields, str):
                metadata_fields = [metadata_fields]

            for k, v in data.items():
                if k in known_fields:
                    continue

                if metadata_fields is None or k in metadata_fields:
                    metadata[k] = v

        return cls(id=id_, text=text_, metadata=metadata or None)

    @staticmethod
    def _apply_rename(data: Dict[str, Any], rename_map: Optional[Dict[str, str]]):
        if not rename_map:
            return data
        return {rename_map.get(k, k): v for k, v in data.items()}
