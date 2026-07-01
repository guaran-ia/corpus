from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from pathlib import Path
from typing import Dict, Iterator, List

from dotenv import load_dotenv
from tqdm import tqdm

from .perplexity_methods import unload_model
from .perplexity_metrics import (
    compute_perplexity_metrics_batch_for_model,
    record_has_metric,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

INPUT_DIR = Path(
    os.getenv(
        "PERPLEXITY_INPUT_DIR",
        str(BASE_DIR / "data" / "processed"),
    )
)

BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        "1",
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

    Args:
        None.

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


def get_metrics_to_process(model: str) -> List[str]:
    """
    Resolve the perplexity metrics that should be processed.

    Args:
        model (str): Selected model name from the command line.

    Returns:
        List[str]: Metadata metric names to compute.
    """
    if model == "all":
        return list(METRIC_STEPS)

    return [
        MODEL_TO_METRIC[model]
    ]


def read_jsonl(path: Path) -> Iterator[Dict]:
    """
    Read a JSONL file and yield one record per non-empty line.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        Iterator[Dict]: Iterator over parsed JSON records.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def extract_text(record: Dict) -> str:
    """
    Extract the document text from a corpus record.

    Args:
        record (Dict): Corpus document record.

    Returns:
        str: Extracted text from text, sentence, or content.
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

    Returns:
        None.
    """
    for record in records:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        pbar.update(1)


def write_batch_for_metric(
    output_file,
    records: List[Dict],
    texts: List[str],
    metric_name: str,
    pbar: tqdm,
) -> None:
    """
    Compute a metric for missing records in a batch and write all records.

    Args:
        output_file: Open writable file object.
        records (List[Dict]): Batch of corpus document records.
        texts (List[str]): Texts extracted from the batch records.
        metric_name (str): Metadata metric to compute.
        pbar (tqdm): Progress bar to update.

    Returns:
        None.
    """
    records_to_compute: List[Dict] = []
    texts_to_compute: List[str] = []
    computed_positions: List[int] = []

    updated_records = [
        dict(record)
        for record in records
    ]

    for index, record in enumerate(records):
        if record_has_metric(
            record,
            metric_name,
        ):
            continue

        records_to_compute.append(record)
        texts_to_compute.append(texts[index])
        computed_positions.append(index)

    if records_to_compute:
        computed_records = compute_perplexity_metrics_batch_for_model(
            records_to_compute,
            texts_to_compute,
            metric_name,
        )

        for original_position, computed_record in zip(
            computed_positions,
            computed_records,
        ):
            updated_records[original_position] = computed_record

    write_records(
        output_file,
        updated_records,
        pbar,
    )


def process_path_for_metric(
    input_path: Path,
    output_path: Path,
    metric_name: str,
    desc: str,
) -> None:
    """
    Process one JSONL path for a single perplexity metric.

    Args:
        input_path (Path): Source JSONL file to read.
        output_path (Path): Temporary JSONL file to write.
        metric_name (str): Metadata metric to compute.
        desc (str): Progress bar description.

    Returns:
        None.
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

                if len(batch_records) >= BATCH_SIZE:
                    write_batch_for_metric(
                        output_file,
                        batch_records,
                        batch_texts,
                        metric_name,
                        pbar,
                    )

                    batch_records = []
                    batch_texts = []

            if batch_records:
                write_batch_for_metric(
                    output_file,
                    batch_records,
                    batch_texts,
                    metric_name,
                    pbar,
                )

    finally:
        pbar.close()


def process_file_for_metric(
    path: Path,
    metric_name: str,
    file_position: int,
    total_files: int,
) -> None:
    """
    Safely process one corpus file for a single perplexity metric.

    Args:
        path (Path): Corpus JSONL file to update.
        metric_name (str): Metadata metric to compute.
        file_position (int): Current file position.
        total_files (int): Total number of files.

    Returns:
        None.
    """
    tmp_path = path.with_suffix(
        path.suffix + f".{metric_name}.tmp"
    )

    desc = (
        f"[{file_position}/{total_files}] "
        f"{path.name} | {metric_name}"
    )

    try:
        process_path_for_metric(
            input_path=path,
            output_path=tmp_path,
            metric_name=metric_name,
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


def process_metric_for_all_files(
    files: List[Path],
    metric_name: str,
) -> None:
    """
    Process all corpus files for one perplexity metric.

    Args:
        files (List[Path]): Corpus JSONL files to update.
        metric_name (str): Metadata metric to compute.

    Returns:
        None.
    """
    print(
        f"\nStarting metric: {metric_name}"
    )

    files_bar = tqdm(
        files,
        desc=f"Processing {metric_name}",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        for index, path in enumerate(
            files_bar,
            start=1,
        ):
            files_bar.set_postfix(
                current=path.name
            )

            process_file_for_metric(
                path=path,
                metric_name=metric_name,
                file_position=index,
                total_files=len(files),
            )

    finally:
        unload_model()


def main() -> None:
    """
    Run perplexity metadata computation for all configured corpus files.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()

    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:
        print(
            f"No JSONL files found in {INPUT_DIR}"
        )
        return

    metrics_to_process = get_metrics_to_process(
        args.model
    )

    print(
        f"Using INPUT_DIR={INPUT_DIR}"
    )
    print(
        f"Using BATCH_SIZE={BATCH_SIZE}"
    )
    print(
        f"Using model={args.model}"
    )

    for metric_name in metrics_to_process:
        process_metric_for_all_files(
            files,
            metric_name,
        )

    print(
        "\nProcessing completed."
    )


if __name__ == "__main__":
    main()