from __future__ import annotations

import re
from typing import Dict, List, Optional

# Supported PII types.
ALLOWED_TYPES = {
    "email",
    "phone",
    "ip",
}

# DataFog engine configuration.
DATAFOG_ENGINE = "smart"

_DATAFOG_READY = False
_DATAFOG_AVAILABLE = False
_DATAFOG_MODULE = None


# ============================================================
# REGEX PATTERNS
# ============================================================

# Email regex.
EMAIL_REGEX = re.compile(
    r"\b"
    r"[A-Za-z0-9._%+\-]+@"
    r"[A-Za-z0-9ÁÉÍÓÚáéíóúÑñÃẼĨÕŨỸãẽĩõũỹ._\-]+"
    r"\.[A-Za-zÁÉÍÓÚáéíóúÑñÃẼĨÕŨỸãẽĩõũỹ]{2,}"
    r"\b",
    re.UNICODE,
)

# IP regex candidate detector.
IP_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

# Phone regex candidate detector.
PHONE_REGEX = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s\-\(\)]{6,}\d)(?!\w)"
)


def normalize_label(raw_label: str) -> Optional[str]:
    """
    Normalize DataFog labels into supported PII types.
    """

    if not raw_label:
        return None

    label = raw_label.strip().lower()

    if "email" in label or "mail" in label:
        return "email"

    if (
        "phone" in label
        or "telephone" in label
        or label == "tel"
    ):
        return "phone"

    if label in {
        "ip",
        "ip_address",
        "ipaddress",
        "ipv4",
    }:
        return "ip"

    return None


def build_detection(
    pii_type: str,
    start: int,
    end: int,
    text: str,
) -> Dict:
    """
    Build a normalized PII detection object.
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
    """

    parts = value.split(".")

    if len(parts) != 4:
        return False

    try:
        return all(
            0 <= int(part) <= 255
            for part in parts
        )

    except ValueError:
        return False


def is_valid_phone(value: str) -> bool:
    """
    Validate a phone candidate and reduce false positives.
    """

    stripped = value.strip()

    # Keep only numeric digits.
    digits = re.sub(
        r"\D",
        "",
        stripped,
    )

    # Reject too short or too long numbers.
    if len(digits) < 8 or len(digits) > 15:
        return False

    # Block common false positive patterns.
    blocked_patterns = (
        r"\d{4}\s*[-/]\s*\d{4}",
        r"\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}",
        r"\d{4}\s*[-/]\s*\d{2,4}",
    )

    if any(
        re.fullmatch(pattern, stripped)
        for pattern in blocked_patterns
    ):
        return False

    return True


def is_valid_detection(
    pii_type: str,
    value: str,
) -> bool:
    """
    Validate a detection according to its PII type.
    """

    if pii_type == "email":
        return (
            EMAIL_REGEX.fullmatch(
                value.strip()
            )
            is not None
        )

    if pii_type == "phone":
        return is_valid_phone(value)

    if pii_type == "ip":
        return is_valid_ip(value)

    return False


def init_datafog() -> None:
    """
    Initialize DataFog only once.
    """

    global _DATAFOG_READY
    global _DATAFOG_AVAILABLE
    global _DATAFOG_MODULE

    # Avoid repeated initialization.
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
    Detect PII candidates using regex patterns.
    """

    detections: List[Dict] = []

    if not text:
        return detections

    # ========================================================
    # EMAIL DETECTION
    # ========================================================

    for match in EMAIL_REGEX.finditer(text):
        detections.append(
            build_detection(
                "email",
                match.start(),
                match.end() - 1,
                match.group(0),
            )
        )

    # ========================================================
    # IP DETECTION
    # ========================================================

    for match in IP_REGEX.finditer(text):

        value = match.group(0)

        if is_valid_ip(value):
            detections.append(
                build_detection(
                    "ip",
                    match.start(),
                    match.end() - 1,
                    value,
                )
            )

    # ========================================================
    # PHONE DETECTION
    # ========================================================

    for match in PHONE_REGEX.finditer(text):

        value = match.group(0)

        if is_valid_phone(value):
            detections.append(
                build_detection(
                    "phone",
                    match.start(),
                    match.end() - 1,
                    value,
                )
            )

    return detections


def detect_pii_datafog(text: str) -> List[Dict]:
    """
    Detect PII candidates using DataFog.
    """

    detections: List[Dict] = []

    if not text:
        return detections

    init_datafog()

    # Stop if DataFog is unavailable.
    if (
        not _DATAFOG_AVAILABLE
        or _DATAFOG_MODULE is None
    ):
        return detections

    try:
        scan = _DATAFOG_MODULE.scan_prompt(
            text,
            engine=DATAFOG_ENGINE,
        )

        entities = getattr(
            scan,
            "entities",
            [],
        ) or []

        for entity in entities:

            # Extract possible label names.
            raw_label = (
                getattr(entity, "entity_type", None)
                or getattr(entity, "type", None)
                or getattr(entity, "label", None)
                or ""
            )

            pii_type = normalize_label(
                str(raw_label)
            )

            # Ignore unsupported labels.
            if pii_type not in ALLOWED_TYPES:
                continue

            start = getattr(
                entity,
                "start",
                None,
            )

            end = getattr(
                entity,
                "end",
                None,
            )

            # Ignore invalid positions.
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
            ):
                continue

            inclusive_end = end - 1

            if inclusive_end < start:
                continue

            value = text[start:end]

            # Apply validation rules.
            if not is_valid_detection(
                pii_type,
                value,
            ):
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


def spans_overlap(
    span_a: Dict,
    span_b: Dict,
) -> bool:
    """
    Check whether two spans overlap.
    """

    return not (
        span_a["end"] < span_b["start"]
        or span_b["end"] < span_a["start"]
    )


def deduplicate_detections(
    detections: List[Dict],
) -> List[Dict]:
    """
    Remove duplicated detections.

    Duplicates are identified using:
    - type
    - start
    - end
    - text
    """

    seen = set()

    deduped: List[Dict] = []

    for detection in sorted(
        detections,
        key=lambda item: (
            item["start"],
            item["end"],
            item["type"],
        ),
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
    Detection rules:

    - email:
        regex only

    - phone:
        regex + DataFog confirmation

    - ip:
        regex + DataFog confirmation
    """

    regex_detections = detect_pii_regex(text)

    if not regex_detections:
        return []

    final_detections: List[Dict] = []

    # ========================================================
    # EMAIL -> REGEX ONLY
    # ========================================================

    for regex_item in regex_detections:

        if regex_item["type"] == "email":
            final_detections.append(
                regex_item
            )

    # ========================================================
    # PHONE + IP -> REQUIRE DATAFOG VALIDATION
    # ========================================================

    regex_items_to_confirm = [
        item
        for item in regex_detections
        if item["type"] in {
            "phone",
            "ip",
        }
    ]

    if not regex_items_to_confirm:
        return deduplicate_detections(
            final_detections
        )

    datafog_detections = detect_pii_datafog(text)

    if not datafog_detections:
        return deduplicate_detections(
            final_detections
        )

    datafog_items_to_confirm = [
        item
        for item in datafog_detections
        if item["type"] in {
            "phone",
            "ip",
        }
    ]

    # ========================================================
    # OVERLAP VALIDATION
    # ========================================================

    for regex_item in regex_items_to_confirm:

        for datafog_item in datafog_items_to_confirm:

            # Both detections must have same type.
            if (
                regex_item["type"]
                != datafog_item["type"]
            ):
                continue

            # Both spans must overlap.
            if spans_overlap(
                regex_item,
                datafog_item,
            ):
                final_detections.append(
                    regex_item
                )
                break

    return deduplicate_detections(
        final_detections
    )