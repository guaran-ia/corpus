from __future__ import annotations

from typing import Dict

from .perplexity_methods import (
    compute_coreguapa_perplexity,
    compute_tweets_perplexity,
)


def compute_perplexity_metrics(
    record: Dict,
    text: str,
) -> Dict:

    updated_record = dict(record)

    metadata = updated_record.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    safe_text = text or ""

    metadata["coreguapa_perplexity"] = (
        compute_coreguapa_perplexity(
            safe_text
        )
    )

    metadata["tweets_perplexity"] = (
        compute_tweets_perplexity(
            safe_text
        )
    )

    updated_record["metadata"] = metadata

    return updated_record