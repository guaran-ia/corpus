from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from .perplexity_methods import (
    get_available_cuda_devices,
    get_device_memory,
    load_models,
    unload_model,
)
from .perplexity_metrics import (
    compute_perplexity_metrics_batch,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

INPUT_DIR = Path(
    os.getenv(
        "PERPLEXITY_INPUT_DIR",
        str(BASE_DIR / "data" / "processed"),
    )
)

DOCUMENT_BATCH_SIZE = int(
    os.getenv(
        "PERPLEXITY_DOCUMENT_BATCH_SIZE",
        os.getenv("BATCH_SIZE", "8"),
    )
)

METRIC_STEPS = (
    "coreguapa_perplexity",
    "tweets_perplexity",
)

MODEL_TO_METRIC = {
    "coreguapa": "coreguapa_perplexity",
    "tweets": "tweets_perplexity",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute corpus perplexity metadata."
    )

    parser.add_argument(
        "--model",
        choices=[
            "coreguapa",
            "tweets",
            "all",
        ],
        required=True,
        help=(
            "Model to run. Use 'coreguapa', 'tweets', "
            "or 'all'."
        ),
    )

    return parser.parse_args()


def validate_runtime_config() -> None:
    """
    Validate pipeline-level configuration.

    Raises:
        ValueError: If DOCUMENT_BATCH_SIZE is invalid.
    """
    if DOCUMENT_BATCH_SIZE <= 0:
        raise ValueError(
            "PERPLEXITY_DOCUMENT_BATCH_SIZE must be greater "
            f"than 0. Current value: {DOCUMENT_BATCH_SIZE}"
        )


def get_metrics_to_process(model: str) -> List[str]:
    """
    Resolve the perplexity metrics that should be processed.

    Args:
        model (str): Selected model name.

    Returns:
        List[str]: Metadata metric names to compute.
    """
    if model == "all":
        return list(METRIC_STEPS)

    return [MODEL_TO_METRIC[model]]


def read_jsonl(path: Path) -> Iterator[Dict]:
    """
    Read a JSONL file and yield one record per non-empty line.

    Args:
        path (Path): Path to the JSONL file.

    Yields:
        Dict: Parsed JSON object.

    Raises:
        ValueError: If a line does not contain a JSON object.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} "
                    f"at line {line_number}."
                )

            yield record


def extract_text(record: Dict) -> str:
    """
    Extract the document text from a corpus record.

    Fields are checked in this order: text, sentence, content.

    Args:
        record (Dict): Corpus document record.

    Returns:
        str: Extracted text or an empty string.
    """
    for key in (
        "text",
        "sentence",
        "content",
    ):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def write_records(
    output_file,
    records: List[Dict],
    pbar: tqdm,
) -> None:
    """
    Write updated records to a JSONL output file.

    Args:
        output_file: Open writable file object.
        records (List[Dict]): Records to write.
        pbar (tqdm): Progress bar to update.
    """
    for record in records:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

        pbar.update(1)


def write_batch_for_metrics(
    output_file,
    records: List[Dict],
    texts: List[str],
    metric_names: Sequence[str],
    concurrent: bool,
    pbar: tqdm,
) -> None:
    """
    Compute missing metrics and write the complete record batch.

    Args:
        output_file: Open writable file object.
        records (List[Dict]): Batch of corpus records.
        texts (List[str]): Texts extracted from the records.
        metric_names (Sequence[str]): Metadata metrics to compute.
        concurrent (bool): Whether models should run concurrently.
        pbar (tqdm): Progress bar to update.
    """
    if len(records) != len(texts):
        raise ValueError(
            "Records and texts must have the same length. "
            f"Records: {len(records)}. "
            f"Texts: {len(texts)}."
        )

    updated_records = compute_perplexity_metrics_batch(
        records=records,
        texts=texts,
        metric_names=metric_names,
        concurrent=concurrent,
    )

    write_records(
        output_file=output_file,
        records=updated_records,
        pbar=pbar,
    )


def process_path_for_metrics(
    input_path: Path,
    output_path: Path,
    metric_names: Sequence[str],
    concurrent: bool,
    desc: str,
) -> None:
    """
    Process one JSONL path for one or more perplexity metrics.

    Args:
        input_path (Path): Source JSONL file.
        output_path (Path): Temporary JSONL output file.
        metric_names (Sequence[str]): Metadata metrics to compute.
        concurrent (bool): Whether models should run concurrently.
        desc (str): Progress bar description.
    """
    pbar = tqdm(
        desc=desc,
        unit="rec",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    batch_records: List[Dict] = []
    batch_texts: List[str] = []

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for record in read_jsonl(input_path):
                batch_records.append(record)
                batch_texts.append(
                    extract_text(record)
                )

                if len(batch_records) >= DOCUMENT_BATCH_SIZE:
                    write_batch_for_metrics(
                        output_file=output_file,
                        records=batch_records,
                        texts=batch_texts,
                        metric_names=metric_names,
                        concurrent=concurrent,
                        pbar=pbar,
                    )

                    batch_records = []
                    batch_texts = []

            if batch_records:
                write_batch_for_metrics(
                    output_file=output_file,
                    records=batch_records,
                    texts=batch_texts,
                    metric_names=metric_names,
                    concurrent=concurrent,
                    pbar=pbar,
                )

    finally:
        pbar.close()


def process_file_for_metrics(
    path: Path,
    metric_names: Sequence[str],
    concurrent: bool,
    file_position: int,
    total_files: int,
) -> None:
    """
    Safely process one corpus file for one or more metrics.

    Args:
        path (Path): Corpus JSONL file to update.
        metric_names (Sequence[str]): Metadata metrics to compute.
        concurrent (bool): Whether models should run concurrently.
        file_position (int): Current file position.
        total_files (int): Total number of files.
    """
    metrics_suffix = ".".join(metric_names)

    tmp_path = path.with_suffix(
        path.suffix + f".{metrics_suffix}.tmp"
    )

    desc = (
        f"[{file_position}/{total_files}] "
        f"{path.name} | {','.join(metric_names)}"
    )

    try:
        process_path_for_metrics(
            input_path=path,
            output_path=tmp_path,
            metric_names=metric_names,
            concurrent=concurrent,
            desc=desc,
        )

        tmp_path.replace(path)

    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()

            except OSError as cleanup_error:
                logging.warning(
                    "Failed to remove temporary file %s: %s",
                    tmp_path,
                    cleanup_error,
                )

        raise


def print_runtime_resources() -> None:
    """
    Print the CUDA resources visible to the process.
    """
    cuda_devices = get_available_cuda_devices()

    if not cuda_devices:
        print("CUDA is not available. Using CPU inference.")
        return

    for device_index in cuda_devices:
        free_memory, total_memory = get_device_memory(
            device_index
        )

        free_gib = free_memory / (1024 ** 3)
        total_gib = total_memory / (1024 ** 3)

        print(
            f"CUDA device {device_index}: "
            f"free={free_gib:.2f} GiB, "
            f"total={total_gib:.2f} GiB"
        )


def process_all_files(
    files: List[Path],
    metric_names: Sequence[str],
    concurrent: bool,
) -> None:
    """
    Process all corpus files for the requested metrics.

    Args:
        files (List[Path]): Corpus JSONL files to update.
        metric_names (Sequence[str]): Metadata metrics to compute.
        concurrent (bool): Whether models should run concurrently.
    """
    print(
        "\nStarting metrics: "
        + ", ".join(metric_names)
    )

    files_bar = tqdm(
        files,
        desc="Processing perplexity metadata",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for index, path in enumerate(
        files_bar,
        start=1,
    ):
        files_bar.set_postfix(
            current=path.name
        )

        process_file_for_metrics(
            path=path,
            metric_names=metric_names,
            concurrent=concurrent,
            file_position=index,
            total_files=len(files),
        )


def can_run_metrics_concurrently(
    metric_names: Sequence[str],
) -> bool:
    """
    Check whether each requested metric can use a separate CUDA device.

    Args:
        metric_names (Sequence[str]): Requested metrics.

    Returns:
        bool: True when concurrent execution is possible.
    """
    cuda_devices = get_available_cuda_devices()

    return (
        len(metric_names) > 1
        and len(cuda_devices) >= len(metric_names)
    )


def run_requested_metrics(
    files: List[Path],
    metric_names: Sequence[str],
    concurrent: bool,
) -> None:
    """
    Process requested metrics concurrently or sequentially.

    Args:
        files (List[Path]): Corpus JSONL files.
        metric_names (Sequence[str]): Requested metrics.
        concurrent (bool): Whether concurrent execution is enabled.
    """
    if concurrent:
        try:
            load_models(metric_names)

            process_all_files(
                files=files,
                metric_names=metric_names,
                concurrent=True,
            )

        finally:
            unload_model()

        return

    for metric_name in metric_names:
        print(f"Running metric sequentially: {metric_name}")
        try:
            load_models([metric_name])

            process_all_files(
                files=files,
                metric_names=[metric_name],
                concurrent=False,
            )

        finally:
            unload_model(metric_name)


def main() -> None:
    """
    Run perplexity metadata computation.
    """
    validate_runtime_config()

    args = parse_args()

    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:
        print(
            f"No JSONL files found in {INPUT_DIR}"
        )
        return

    metric_names = get_metrics_to_process(
        args.model
    )

    concurrent = (
        args.model == "all"
        and can_run_metrics_concurrently(
            metric_names
        )
    )

    print(
        f"Using INPUT_DIR={INPUT_DIR}"
    )
    print(
        "Using PERPLEXITY_DOCUMENT_BATCH_SIZE="
        f"{DOCUMENT_BATCH_SIZE}"
    )
    print(
        f"Using model={args.model}"
    )
    print(
        f"Concurrent model execution={concurrent}"
    )

    print_runtime_resources()

    run_requested_metrics(
        files=files,
        metric_names=metric_names,
        concurrent=concurrent,
    )

    print(
        "\nProcessing completed."
    )


if __name__ == "__main__":
    main()
