import unicodedata
import re

def normalize_guarani(text: str) -> str:
    """
    Normalize Guaraní text by removing diacritics (acute accents and nasal tildes)
    and glottal stop apostrophes, to allow approximate string matching.
    """
    text = unicodedata.normalize("NFD", text)

    # Remove diacritics (accents and nasal tildes)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # Remove glottal stop apostrophes
    text = re.sub(r"[ʼ’']", "", text)

    return text.lower()
