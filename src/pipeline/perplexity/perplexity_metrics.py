from __future__ import annotations

from typing import Dict, List

from .perplexity_methods import compute_perplexity_batch_for_metric


def record_has_metric(record: Dict, metric_name: str) -> bool:
    """
    Check whether a record already contains a given perplexity metric.

    Args:
        record (Dict): Corpus document record.
        metric_name (str): Metadata metric name to check.

    Returns:
        bool: True if the metadata exists and is not None, otherwise False.
    """
    metadata = record.get("metadata", {})

    return (
        isinstance(metadata, dict)
        and metric_name in metadata
        and metadata[metric_name] is not None
    )


def compute_perplexity_metrics_batch_for_model(
    records: List[Dict],
    texts: List[str],
    metric_name: str,
) -> List[Dict]:
    """
    Compute one perplexity metric and add it to a batch of records.

    Args:
        records (List[Dict]): Corpus document records to update.
        texts (List[str]): Texts extracted from the records.
        metric_name (str): Metadata metric to compute and store.

    Returns:
        List[Dict]: Updated records with the computed metric inside metadata.
    """
    safe_texts = [
        text or ""
        for text in texts
    ]

    values = compute_perplexity_batch_for_metric(
        metric_name,
        safe_texts,
    )

    updated_records: List[Dict] = []

    for record, value in zip(
        records,
        values,
    ):
        updated_record = dict(record)

        metadata = updated_record.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        else:
            metadata = dict(metadata)

        metadata[metric_name] = value

        updated_record["metadata"] = metadata
        updated_records.append(updated_record)

    return updated_records