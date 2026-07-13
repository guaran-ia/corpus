from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from pathlib import Path
from typing import Dict, Iterator, List

from tqdm import tqdm


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent
)

INPUT_DIR = Path(
    os.getenv(
        "PERPLEXITY_INPUT_DIR",
        str(
            BASE_DIR
            / "data"
            / "processed"
        ),
    )
)

PERPLEXITY_METRICS = (
    "coreguapa_perplexity",
    "tweets_perplexity",
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace:
            Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove perplexity metrics from processed JSONL files "
            "so they can be recomputed."
        )
    )

    parser.add_argument(
        "--metric",
        choices=[
            "coreguapa",
            "tweets",
            "all",
        ],
        default="all",
        help=(
            "Metric to remove. Use 'coreguapa', 'tweets', "
            "or 'all'. Default: all."
        ),
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help=(
            "Create a .bak copy of each original JSONL file "
            "before replacing it."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Count the metrics that would be removed without "
            "modifying any files."
        ),
    )

    return parser.parse_args()


def get_metrics_to_remove(
    selected_metric: str,
) -> List[str]:
    """
    Resolve the metric fields that must be removed.

    Args:
        selected_metric (str):
            Metric option selected on the command line.

    Returns:
        List[str]:
            JSON field names to remove.
    """
    if selected_metric == "coreguapa":
        return [
            "coreguapa_perplexity"
        ]

    if selected_metric == "tweets":
        return [
            "tweets_perplexity"
        ]

    return list(
        PERPLEXITY_METRICS
    )


def read_jsonl(
    path: Path,
) -> Iterator[tuple[int, Dict]]:
    """
    Read a JSONL file and yield one object per non-empty line.

    Args:
        path (Path):
            Path to the JSONL file.

    Yields:
        tuple[int, Dict]:
            Line number and parsed JSON object.

    Raises:
        ValueError:
            If a line does not contain a JSON object.
        json.JSONDecodeError:
            If a line contains invalid JSON.
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

            record = json.loads(
                line
            )

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Expected a JSON object in {path} "
                    f"at line {line_number}."
                )

            yield (
                line_number,
                record,
            )


def remove_metrics_from_record(
    record: Dict,
    metric_names: List[str],
) -> int:
    """
    Remove selected perplexity metrics from one document.

    Args:
        record (Dict):
            Corpus document record.
        metric_names (List[str]):
            Metric fields to remove.

    Returns:
        int:
            Number of fields removed from the record.
    """
    removed_count = 0

    for metric_name in metric_names:
        if metric_name in record:
            record.pop(
                metric_name,
                None,
            )

            removed_count += 1

    return removed_count


def inspect_file(
    path: Path,
    metric_names: List[str],
) -> tuple[int, int]:
    """
    Count records and removable metric fields without modifying a file.

    Args:
        path (Path):
            JSONL file to inspect.
        metric_names (List[str]):
            Metric fields to count.

    Returns:
        tuple[int, int]:
            Number of records and number of metric fields found.
    """
    record_count = 0
    removed_field_count = 0

    for _, record in read_jsonl(
        path
    ):
        record_count += 1

        removed_field_count += sum(
            1
            for metric_name in metric_names
            if metric_name in record
        )

    return (
        record_count,
        removed_field_count,
    )


def process_file(
    path: Path,
    metric_names: List[str],
    create_backup: bool,
) -> tuple[int, int]:
    """
    Remove selected metrics from one JSONL file safely.

    The updated file is first written to a temporary path. The original
    file is replaced only after the complete JSONL file has been processed
    successfully.

    Args:
        path (Path):
            JSONL file to update.
        metric_names (List[str]):
            Metric fields to remove.
        create_backup (bool):
            Whether to preserve a .bak copy of the original file.

    Returns:
        tuple[int, int]:
            Number of records processed and number of metric fields removed.
    """
    temporary_path = path.with_suffix(
        path.suffix + ".remove_perplexity.tmp"
    )

    backup_path = path.with_suffix(
        path.suffix + ".bak"
    )

    record_count = 0
    removed_field_count = 0

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for _, record in read_jsonl(
                path
            ):
                record_count += 1

                removed_field_count += (
                    remove_metrics_from_record(
                        record=record,
                        metric_names=metric_names,
                    )
                )

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if create_backup:
            shutil.copy2(
                path,
                backup_path,
            )

        temporary_path.replace(
            path
        )

    except BaseException:
        if temporary_path.exists():
            try:
                temporary_path.unlink()

            except OSError:
                pass

        raise

    return (
        record_count,
        removed_field_count,
    )


def main() -> None:
    """
    Remove selected perplexity metrics from all processed corpora.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit:
            If no JSONL files are found.
    """
    args = parse_args()

    metric_names = get_metrics_to_remove(
        args.metric
    )

    files = sorted(
        INPUT_DIR.glob(
            "*.jsonl"
        )
    )

    if not files:
        raise SystemExit(
            f"No JSONL files found in {INPUT_DIR}."
        )

    print(
        f"Using INPUT_DIR={INPUT_DIR}"
    )
    print(
        "Metrics to remove="
        + ", ".join(
            metric_names
        )
    )
    print(
        f"Dry run={args.dry_run}"
    )
    print(
        f"Create backups={args.backup}"
    )

    total_records = 0
    total_removed_fields = 0

    files_bar = tqdm(
        files,
        desc=(
            "Inspecting perplexity metadata"
            if args.dry_run
            else "Removing perplexity metadata"
        ),
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for path in files_bar:
        files_bar.set_postfix(
            current=path.name
        )

        if args.dry_run:
            (
                record_count,
                removed_field_count,
            ) = inspect_file(
                path=path,
                metric_names=metric_names,
            )

        else:
            (
                record_count,
                removed_field_count,
            ) = process_file(
                path=path,
                metric_names=metric_names,
                create_backup=args.backup,
            )

        total_records += record_count
        total_removed_fields += (
            removed_field_count
        )

    print()
    print(
        "=" * 80
    )
    print(
        "Perplexity metadata removal summary"
    )
    print(
        "=" * 80
    )
    print(
        f"Files processed       : {len(files)}"
    )
    print(
        f"Documents processed   : {total_records}"
    )
    print(
        f"Metric fields found   : {total_removed_fields}"
    )

    if args.dry_run:
        print(
            "No files were modified."
        )
    else:
        print(
            f"Metric fields removed : {total_removed_fields}"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
