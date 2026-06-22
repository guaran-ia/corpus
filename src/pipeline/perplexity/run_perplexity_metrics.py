from __future__ import annotations

import json
import logging
import sys

from pathlib import Path
from typing import Dict, Iterator, List

from tqdm import tqdm

from .perplexity_metrics import (
    compute_perplexity_metrics,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent
)

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


def read_jsonl(
    path: Path,
) -> Iterator[Dict]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                yield json.loads(line)


def extract_text(
    record: Dict,
) -> str:

    for key in (
        "text",
        "sentence",
        "content",
    ):
        value = record.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value

    return ""


def init_file_report(
    file_path: Path,
) -> Dict:

    return {
        "file_name": file_path.name,
        "total_records": 0,
        "records_with_coreguapa_perplexity": 0,
        "records_with_tweets_perplexity": 0,
    }


def update_file_report(
    report: Dict,
    updated_record: Dict,
) -> None:

    report["total_records"] += 1

    metadata = updated_record.get(
        "metadata",
        {},
    )

    if "coreguapa_perplexity" in metadata:
        report[
            "records_with_coreguapa_perplexity"
        ] += 1

    if "tweets_perplexity" in metadata:
        report[
            "records_with_tweets_perplexity"
        ] += 1


def process_file(
    path: Path,
    file_position: int,
    total_files: int,
) -> Dict:

    report = init_file_report(
        path
    )

    desc = (
        f"[{file_position}/{total_files}] "
        f"{path.name}"
    )

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    pbar = tqdm(
        desc=desc,
        unit="rec",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        with tmp_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for index, record in enumerate(
                read_jsonl(path),
                start=1,
            ):

                text = extract_text(record)

                updated_record = compute_perplexity_metrics(
                    record,
                    text,
                )

                output_file.write(
                    json.dumps(
                        updated_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                update_file_report(
                    report,
                    updated_record,
                )

                pbar.update(1)

                if index % 500 == 0:
                    pbar.set_postfix(
                        processed=index,
                        coreguapa=report[
                            "records_with_coreguapa_perplexity"
                        ],
                        tweets=report[
                            "records_with_tweets_perplexity"
                        ],
                    )

        tmp_path.replace(path)

        return report

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

    finally:
        pbar.close()


def main() -> None:

    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:
        print(
            f"No JSONL files found in "
            f"{INPUT_DIR}"
        )

        return

    reports: List[Dict] = []

    files_bar = tqdm(
        files,
        desc="Processing files",
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

        report = process_file(
            path,
            index,
            len(files),
        )

        reports.append(report)

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()