from __future__ import annotations

from typing import Dict, List

from .pii_methods import detect_pii_combined

MAX_CHUNK_LENGTH = 1000


def split_text_into_chunks(
    text: str,
    max_length: int = MAX_CHUNK_LENGTH,
) -> List[Dict]:

    text = text or ""

    if not text:
        return []

    if len(text) <= max_length:
        return [
            {
                "text": text,
                "offset": 0,
            }
        ]

    chunks: List[Dict] = []

    text_length = len(text)

    start = 0

    while start < text_length:
        end = min(
            start + max_length,
            text_length,
        )

        if (
            end < text_length
            and not text[end].isspace()
        ):
            last_space = text.rfind(
                " ",
                start,
                end,
            )

            last_newline = text.rfind(
                "\n",
                start,
                end,
            )

            split_at = max(
                last_space,
                last_newline,
            )

            if split_at > start:
                end = split_at

        chunk_text = text[start:end].strip()

        if chunk_text:
            real_start = start

            while (
                real_start < text_length
                and text[real_start].isspace()
            ):
                real_start += 1

            chunks.append(
                {
                    "text": chunk_text,
                    "offset": real_start,
                }
            )

        start = end

        while (
            start < text_length
            and text[start].isspace()
        ):
            start += 1

    return chunks


def adjust_chunk_detections(
    detections: List[Dict],
    offset: int,
) -> List[Dict]:

    adjusted: List[Dict] = []

    for detection in detections:
        item = detection.copy()

        item["start"] += offset
        item["end"] += offset

        adjusted.append(item)

    return adjusted


def deduplicate_final_spans(
    spans: List[Dict],
) -> List[Dict]:

    if not spans:
        return []

    seen = set()

    deduped: List[Dict] = []

    for span in sorted(
        spans,
        key=lambda item: (
            item["start"],
            item["end"],
            item["type"],
        ),
    ):
        key = (
            span["type"],
            span["start"],
            span["end"],
            span.get("text", ""),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(span)

    return deduped


def merge_spans(
    spans: List[Dict],
) -> List[Dict]:

    if not spans:
        return []

    ordered = sorted(
        spans,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    merged = [
        {
            "start": ordered[0]["start"],
            "end": ordered[0]["end"],
        }
    ]

    for span in ordered[1:]:
        current = merged[-1]

        if (
            span["start"]
            <= current["end"] + 1
        ):
            current["end"] = max(
                current["end"],
                span["end"],
            )

        else:
            merged.append(
                {
                    "start": span["start"],
                    "end": span["end"],
                }
            )

    return merged


def compute_pii_proportion(
    text: str,
    spans: List[Dict],
) -> float:

    if not text:
        return 0.0

    merged = merge_spans(spans)

    covered_chars = sum(
        span["end"]
        - span["start"]
        + 1
        for span in merged
    )

    return covered_chars / len(text)


def compute_pii_metrics(
    record: Dict,
    text: str,
) -> Dict:

    updated_record = dict(record)

    safe_text = text or ""

    if not safe_text.strip():
        updated_record["has_pii"] = False
        updated_record["pii_prop"] = 0.0
        updated_record["pii_spans"] = []

        return updated_record

    all_detections: List[Dict] = []

    chunks = split_text_into_chunks(
        safe_text,
        max_length=MAX_CHUNK_LENGTH,
    )

    for chunk in chunks:
        chunk_text = chunk["text"]

        chunk_offset = chunk["offset"]

        chunk_detections = detect_pii_combined(
            chunk_text
        )

        chunk_detections = adjust_chunk_detections(
            chunk_detections,
            chunk_offset,
        )

        all_detections.extend(
            chunk_detections
        )

    all_detections = deduplicate_final_spans(
        all_detections
    )

    all_detections = sorted(
        all_detections,
        key=lambda item: (
            item["start"],
            item["end"],
            item["type"],
        ),
    )

    updated_record["has_pii"] = bool(
        all_detections
    )

    updated_record["pii_prop"] = (
        compute_pii_proportion(
            safe_text,
            all_detections,
        )
    )

    updated_record["pii_spans"] = all_detections

    return updated_record