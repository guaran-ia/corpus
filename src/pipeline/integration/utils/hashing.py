import hashlib

def make_doc_id(text: str) -> str:
    """
    Create a deterministic ID for a document based on its text.
    Duplicate documents will produce the same ID.
    """

    text = text.strip()

    return hashlib.sha256(text.encode("utf-8")).hexdigest()