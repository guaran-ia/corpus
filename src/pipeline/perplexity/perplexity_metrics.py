from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Sequence

from .perplexity_methods import (
    compute_perplexity_batch_for_metric,
)


def record_has_metric(
    record: Dict,
    metric_name: str,
) -> bool:
    """
    Check whether a record already contains a perplexity metric.

    The metric is expected at the root level of the record, for example:

        {
            "text": "...",
            "coreguapa_perplexity": 12.45
        }

    Args:
        record (Dict): Corpus document record.
        metric_name (str): Metric name.

    Returns:
        bool: True if the metric exists and its value is not None.
    """
    return record.get(metric_name) is not None


def compute_perplexity_values_for_metrics(
    texts_by_metric: Dict[str, List[str]],
    concurrent: bool,
) -> Dict[str, List[float | None]]:
    """
    Compute one or more perplexity metrics for independent text batches.

    When concurrent is enabled, each model runs in its own worker and CUDA
    stream. This allows both resident models to execute at the same time when
    the runtime and GPU scheduler have enough resources.

    Args:
        texts_by_metric (Dict[str, List[str]]): Texts grouped by metric.
        concurrent (bool): Whether metrics should run concurrently.

    Returns:
        Dict[str, List[float | None]]: Perplexity values grouped by metric.
    """
    if not texts_by_metric:
        return {}

    if not concurrent or len(texts_by_metric) == 1:
        return {
            metric_name: compute_perplexity_batch_for_metric(
                metric_name=metric_name,
                texts=texts,
            )
            for metric_name, texts in texts_by_metric.items()
        }

    results: Dict[str, List[float | None]] = {}

    with ThreadPoolExecutor(
        max_workers=len(texts_by_metric),
        thread_name_prefix="perplexity-model",
    ) as executor:
        futures: Dict[str, Future] = {
            metric_name: executor.submit(
                compute_perplexity_batch_for_metric,
                metric_name,
                texts,
            )
            for metric_name, texts in texts_by_metric.items()
        }

        for metric_name, future in futures.items():
            results[metric_name] = future.result()

    return results


def compute_perplexity_metrics_batch(
    records: List[Dict],
    texts: List[str],
    metric_names: Sequence[str],
    concurrent: bool = False,
) -> List[Dict]:
    """
    Compute missing perplexity metrics and add them to a record batch.

    Each metric is computed only for records where it is absent. When more
    than one metric is requested, model computations can run concurrently.

    Args:
        records (List[Dict]): Corpus records to update.
        texts (List[str]): Texts extracted from the records.
        metric_names (Sequence[str]): Perplexity metrics to compute.
        concurrent (bool): Whether different models should run concurrently.

    Returns:
        List[Dict]: Updated records.

    Raises:
        ValueError: If records and texts have different lengths.
        RuntimeError: If the number of metric values is incorrect.
    """
    if len(records) != len(texts):
        raise ValueError(
            "Records and texts must have the same length. "
            f"Records: {len(records)}. "
            f"Texts: {len(texts)}."
        )

    safe_texts = [
        text if isinstance(text, str) else ""
        for text in texts
    ]

    unique_metric_names = list(
        dict.fromkeys(metric_names)
    )

    updated_records = [
        dict(record)
        for record in records
    ]

    positions_by_metric: Dict[str, List[int]] = {}
    texts_by_metric: Dict[str, List[str]] = {}

    for metric_name in unique_metric_names:
        positions = [
            index
            for index, record in enumerate(records)
            if not record_has_metric(record, metric_name)
        ]

        if not positions:
            continue

        positions_by_metric[metric_name] = positions
        texts_by_metric[metric_name] = [
            safe_texts[index]
            for index in positions
        ]

    values_by_metric = compute_perplexity_values_for_metrics(
        texts_by_metric=texts_by_metric,
        concurrent=concurrent,
    )

    for metric_name, positions in positions_by_metric.items():
        values = values_by_metric[metric_name]

        if len(values) != len(positions):
            raise RuntimeError(
                "The number of computed perplexity values does not "
                "match the number of records requiring the metric. "
                f"Metric: {metric_name}. "
                f"Values: {len(values)}. "
                f"Records: {len(positions)}."
            )

        for position, value in zip(
            positions,
            values,
        ):
            updated_records[position][metric_name] = value

    return updated_records


def compute_perplexity_metrics_batch_for_model(
    records: List[Dict],
    texts: List[str],
    metric_name: str,
) -> List[Dict]:
    """
    Compute one perplexity metric and add it to a batch of records.

    This compatibility wrapper preserves the previous public function while
    delegating to the multi-model batch implementation.

    Args:
        records (List[Dict]): Corpus records to update.
        texts (List[str]): Texts extracted from the records.
        metric_name (str): Perplexity metric to compute.

    Returns:
        List[Dict]: Updated records.
    """
    return compute_perplexity_metrics_batch(
        records=records,
        texts=texts,
        metric_names=[metric_name],
        concurrent=False,
    )
