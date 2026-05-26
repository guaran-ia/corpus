from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import time
import warnings

from pathlib import Path
from typing import Dict, Iterator, List

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

os.environ.setdefault(
    "HF_HUB_DISABLE_TELEMETRY",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_VERBOSITY",
    "error",
)

os.environ.setdefault(
    "HF_HUB_VERBOSITY",
    "error",
)

os.environ.setdefault(
    "ORT_LOGGING_LEVEL",
    "4",
)

from tqdm import tqdm

with open(os.devnull, "w") as _devnull:
    with contextlib.redirect_stdout(
        _devnull
    ), contextlib.redirect_stderr(
        _devnull
    ):

        from .pii_metrics import (
            compute_pii_metrics,
        )

        from .pii_methods import (
            init_datafog,
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

REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "report"
)

REPORT_PATH = (
    REPORT_DIR
    / "pii_report.json"
)


def configure_runtime() -> None:

    warnings.filterwarnings(
        "ignore",
        message=".*resume_download.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*Sentence of length.*truncated.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*GetPciBusId.*",
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="huggingface_hub",
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="gliner",
    )

    logging.getLogger(
        "huggingface_hub"
    ).setLevel(logging.ERROR)

    logging.getLogger(
        "transformers"
    ).setLevel(logging.ERROR)

    logging.getLogger(
        "onnxruntime"
    ).setLevel(logging.ERROR)


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


def write_jsonl(
    path: Path,
    records: List[Dict],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def init_file_report(
    file_path: Path,
    total_records: int,
) -> Dict:

    return {
        "file_name": file_path.name,
        "total_records": total_records,
        "records_with_pii": 0,
        "records_without_pii": 0,
        "total_pii_spans": 0,
        "pii_counts_by_type": {
            "email": 0,
            "phone": 0,
            "ip": 0,
        },
    }


def update_file_report(
    report: Dict,
    updated_record: Dict,
) -> None:

    spans = (
        updated_record.get(
            "pii_spans",
            [],
        )
        or []
    )

    if spans:
        report["records_with_pii"] += 1

    else:
        report["records_without_pii"] += 1

    report["total_pii_spans"] += len(spans)

    for span in spans:
        pii_type = span.get("type")

        if (
            pii_type
            in report[
                "pii_counts_by_type"
            ]
        ):
            report[
                "pii_counts_by_type"
            ][pii_type] += 1


def build_global_summary(
    reports: List[Dict],
) -> Dict:

    total_files = len(reports)

    total_records = sum(
        report["total_records"]
        for report in reports
    )

    records_with_pii = sum(
        report["records_with_pii"]
        for report in reports
    )

    pii_counts_by_type = {
        "email": 0,
        "phone": 0,
        "ip": 0,
    }

    for report in reports:
        for (
            pii_type,
            count,
        ) in report[
            "pii_counts_by_type"
        ].items():

            pii_counts_by_type[
                pii_type
            ] += count

    return {
        "total_files": total_files,
        "total_records": total_records,
        "records_with_pii": records_with_pii,
        "records_with_pii_percentage": round(
            (
                records_with_pii
                / total_records
            )
            * 100,
            2,
        )
        if total_records
        else 0.0,
        "pii_counts_by_type": (
            pii_counts_by_type
        ),
    }


def save_report(
    report_path: Path,
    reports: List[Dict],
) -> None:

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = build_global_summary(
        reports
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )


def init_datafog_clean() -> None:

    print(
        "Inicializando DataFog...",
        end="",
        flush=True,
    )

    start = time.time()

    buffer = io.StringIO()

    with contextlib.redirect_stdout(
        buffer
    ), contextlib.redirect_stderr(
        buffer
    ):

        init_datafog()

    elapsed = time.time() - start

    print(f" OK ({elapsed:.2f}s)")


def process_file(
    path: Path,
    file_position: int,
    total_files: int,
) -> Dict:

    total_records = sum(
        1 for _ in read_jsonl(path)
    )

    report = init_file_report(
        path,
        total_records,
    )

    desc = (
        f"[{file_position}/{total_files}] "
        f"{path.name}"
    )

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    pbar = tqdm(
        total=total_records,
        desc=desc,
        unit="rec",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, record in enumerate(
            read_jsonl(path),
            start=1,
        ):

            text = extract_text(record)

            updated_record = compute_pii_metrics(
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

            if (
                index % 500 == 0
                or index == total_records
            ):
                pbar.set_postfix(
                    processed=index,
                    spans=report[
                        "total_pii_spans"
                    ],
                    with_pii=report[
                        "records_with_pii"
                    ],
                )

    pbar.close()

    tmp_path.replace(path)

    return report


def main() -> None:

    configure_runtime()

    init_datafog_clean()

    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:
        print(
            f"No JSONL files found in "
            f"{INPUT_DIR}"
        )

        return

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    save_report(
        REPORT_PATH,
        reports,
    )

    print("\nProcessing completed.")

    print(
        f"Report saved in: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()