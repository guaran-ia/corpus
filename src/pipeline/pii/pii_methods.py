from __future__ import annotations

import re
from typing import Dict, List, Optional

ALLOWED_TYPES = {"email", "phone", "ip", "physical_address"}
DATAFOG_ENGINE = "smart"

_DATAFOG_READY = False
_DATAFOG_AVAILABLE = False
_DATAFOG_MODULE = None


EMAIL_REGEX = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    re.UNICODE,
)

IP_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

PHONE_REGEX = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s\-\(\)]{6,}\d)(?!\w)"
)

ADDRESS_REGEX = re.compile(
    r"\b(?:"
    r"calle|avda\.?|avenida|av\.?|ruta|km|barrio|pasaje|esquina|"
    r"manzana|mz\.?|casa|nro\.?|número|edificio|piso|departamento|dpto\.?"
    r")\s+"
    r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9][A-Za-zÁÉÍÓÚáéíóúÑñ0-9\s\.\-]{2,60}",
    re.IGNORECASE | re.UNICODE,
)


def normalize_label(raw_label: str) -> Optional[str]:
    """
    Normalize a detector label to one supported PII type.

    Args:
        raw_label: Raw label returned by DataFog.

    Returns:
        Normalized PII type if supported, otherwise None.
    """
    if not raw_label:
        return None

    label = raw_label.strip().lower()

    if "email" in label or "mail" in label:
        return "email"

    if "phone" in label or "telephone" in label or "tel" == label:
        return "phone"

    if label in {"ip", "ip_address", "ipaddress", "ipv4"}:
        return "ip"

    if "address" in label or "location" in label:
        return "physical_address"

    return None


def build_detection(pii_type: str, start: int, end: int, text: str) -> Dict:
    """
    Build a normalized detection.

    Args:
        pii_type: PII type.
        start: Start position, base 0.
        end: End position, inclusive.
        text: Detected text.

    Returns:
        Normalized detection dictionary.
    """
    return {
        "type": pii_type,
        "start": start,
        "end": end,
        "text": text,
    }


def is_valid_ip(value: str) -> bool:
    """
    Validate an IPv4 candidate.

    Args:
        value: Candidate IP.

    Returns:
        True if valid IPv4, otherwise False.
    """
    parts = value.split(".")
    if len(parts) != 4:
        return False

    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def is_valid_phone(value: str) -> bool:
    """
    Validate a phone candidate and reduce false positives.

    Args:
        value: Candidate phone.

    Returns:
        True if the candidate looks like a phone number.
    """
    stripped = value.strip()
    digits = re.sub(r"\D", "", stripped)

    if len(digits) < 8 or len(digits) > 15:
        return False

    blocked_patterns = (
        r"\d{4}\s*[-/]\s*\d{4}",
        r"\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}",
        r"\d{4}\s*[-/]\s*\d{2,4}",
    )

    if any(re.fullmatch(pattern, stripped) for pattern in blocked_patterns):
        return False

    return True


def is_valid_physical_address(value: str) -> bool:
    """
    Validate a physical address candidate.

    Args:
        value: Candidate address.

    Returns:
        True if the candidate looks like a physical address.
    """
    lowered = value.lower().strip()

    blocked = (
        "dirección de correo",
        "correo electrónico",
        "email address",
        "número de teléfono",
        "guive",
        "peve",
        "gotyo",
        "rupi",
        "jerére",
        "ciudad",
        "isla",
        "mundo",
    )

    if any(item in lowered for item in blocked):
        return False

    keywords = (
        "calle",
        "av",
        "avenida",
        "ruta",
        "km",
        "pasaje",
        "esquina",
    )

    has_keyword = any(keyword in lowered for keyword in keywords)
    has_digit = any(char.isdigit() for char in value)

    if has_keyword and has_digit:
        return True

    if has_keyword and len(value.split()) >= 3:
        return True

    return False


def is_valid_detection(pii_type: str, value: str) -> bool:
    """
    Validate a detection according to its PII type.

    Args:
        pii_type: PII type.
        value: Candidate text.

    Returns:
        True if valid, otherwise False.
    """
    if pii_type == "email":
        return EMAIL_REGEX.fullmatch(value.strip()) is not None

    if pii_type == "phone":
        return is_valid_phone(value)

    if pii_type == "ip":
        return is_valid_ip(value)

    if pii_type == "physical_address":
        return is_valid_physical_address(value)

    return False


def init_datafog() -> None:
    """
    Initialize DataFog only once.

    Returns:
        None.
    """
    global _DATAFOG_READY, _DATAFOG_AVAILABLE, _DATAFOG_MODULE

    if _DATAFOG_READY:
        return

    _DATAFOG_READY = True

    try:
        import datafog  # type: ignore

        _DATAFOG_MODULE = datafog
        _DATAFOG_AVAILABLE = True
    except Exception:
        _DATAFOG_MODULE = None
        _DATAFOG_AVAILABLE = False


def detect_pii_regex(text: str) -> List[Dict]:
    """
    Detect PII candidates using regex.

    Args:
        text: Input text.

    Returns:
        List of regex detections.
    """
    detections: List[Dict] = []

    if not text:
        return detections

    for match in EMAIL_REGEX.finditer(text):
        detections.append(
            build_detection("email", match.start(), match.end() - 1, match.group(0))
        )

    for match in IP_REGEX.finditer(text):
        value = match.group(0)
        if is_valid_ip(value):
            detections.append(
                build_detection("ip", match.start(), match.end() - 1, value)
            )

    for match in PHONE_REGEX.finditer(text):
        value = match.group(0)
        if is_valid_phone(value):
            detections.append(
                build_detection("phone", match.start(), match.end() - 1, value)
            )

    for match in ADDRESS_REGEX.finditer(text):
        value = match.group(0)
        if is_valid_physical_address(value):
            detections.append(
                build_detection(
                    "physical_address",
                    match.start(),
                    match.end() - 1,
                    value,
                )
            )

    return detections


def detect_pii_datafog(text: str) -> List[Dict]:
    """
    Detect PII candidates using DataFog.

    Args:
        text: Input text.

    Returns:
        List of DataFog detections.
    """
    detections: List[Dict] = []

    if not text:
        return detections

    init_datafog()

    if not _DATAFOG_AVAILABLE or _DATAFOG_MODULE is None:
        return detections

    try:
        scan = _DATAFOG_MODULE.scan_prompt(text, engine=DATAFOG_ENGINE)
        entities = getattr(scan, "entities", []) or []

        for entity in entities:
            raw_label = (
                getattr(entity, "entity_type", None)
                or getattr(entity, "type", None)
                or getattr(entity, "label", None)
                or ""
            )

            pii_type = normalize_label(str(raw_label))
            if pii_type not in ALLOWED_TYPES:
                continue

            start = getattr(entity, "start", None)
            end = getattr(entity, "end", None)

            if not isinstance(start, int) or not isinstance(end, int):
                continue

            inclusive_end = end - 1
            if inclusive_end < start:
                continue

            value = text[start:end]

            if not is_valid_detection(pii_type, value):
                continue

            detections.append(
                build_detection(
                    pii_type=pii_type,
                    start=start,
                    end=inclusive_end,
                    text=value,
                )
            )

    except Exception:
        return []

    return detections


def spans_overlap(span_a: Dict, span_b: Dict) -> bool:
    """
    Check whether two spans overlap.

    Args:
        span_a: First span.
        span_b: Second span.

    Returns:
        True if the spans overlap.
    """
    return not (span_a["end"] < span_b["start"] or span_b["end"] < span_a["start"])


def deduplicate_detections(detections: List[Dict]) -> List[Dict]:
    """
    Remove duplicated detections.

    Args:
        detections: Detection list.

    Returns:
        Deduplicated detections.
    """
    seen = set()
    deduped: List[Dict] = []

    for detection in sorted(
        detections,
        key=lambda item: (item["start"], item["end"], item["type"]),
    ):
        key = (
            detection["type"],
            detection["start"],
            detection["end"],
            detection["text"],
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(detection)

    return deduped


def detect_pii_combined(text: str) -> List[Dict]:
    """
    Detect PII using strict regex + DataFog confirmation.

    A detection is kept only when:
    - regex detects a candidate,
    - DataFog detects the same PII type,
    - both spans overlap.

    Args:
        text: Input text.

    Returns:
        Final confirmed detections.
    """
    regex_detections = detect_pii_regex(text)

    if not regex_detections:
        return []

    datafog_detections = detect_pii_datafog(text)

    if not datafog_detections:
        return []

    confirmed: List[Dict] = []

    for regex_item in regex_detections:
        for datafog_item in datafog_detections:
            if regex_item["type"] != datafog_item["type"]:
                continue

            if spans_overlap(regex_item, datafog_item):
                confirmed.append(regex_item)
                break

    return deduplicate_detections(confirmed)