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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")

from tqdm import tqdm

with open(os.devnull, "w") as _devnull:
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        from .pii_metrics import compute_pii_metrics
        from .pii_methods import init_datafog

INPUT_DIR = Path("data/processed")
REPORT_DIR = Path("/home/raraujo/corpus/outputs/report")
REPORT_PATH = REPORT_DIR / "pii_report.json"


def configure_runtime() -> None:
    """
    Configure warnings and logging to keep console output clean.

    Returns:
        None.
    """
    warnings.filterwarnings("ignore", message=".*resume_download.*")
    warnings.filterwarnings("ignore", message=".*Sentence of length.*truncated.*")
    warnings.filterwarnings("ignore", message=".*GetPciBusId.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=UserWarning, module="gliner")

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)


def read_jsonl(path: Path) -> Iterator[Dict]:
    """
    Read a JSONL file line by line.

    Args:
        path: Input JSONL path.

    Yields:
        Parsed JSON records.
    """
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def extract_text(record: Dict) -> str:
    """
    Extract the main text field from a record.

    Args:
        record: Input JSON record.

    Returns:
        Extracted text if found, otherwise empty string.
    """
    for key in ("text", "sentence", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def write_jsonl(path: Path, records: List[Dict]) -> None:
    """
    Write records to a JSONL file.

    Args:
        path: Output JSONL path.
        records: Records to write.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_file_report(file_path: Path, total_records: int) -> Dict:
    """
    Create the initial report structure for one file.

    Args:
        file_path: Input file path.
        total_records: Number of records in the file.

    Returns:
        Initialized report dictionary.
    """
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
            "physical_address": 0,
        },
    }


def update_file_report(report: Dict, updated_record: Dict) -> None:
    """
    Update one file report using one processed record.

    Args:
        report: File report.
        updated_record: Processed record with PII metadata.

    Returns:
        None.
    """
    spans = updated_record.get("pii_spans", []) or []

    if spans:
        report["records_with_pii"] += 1
    else:
        report["records_without_pii"] += 1

    report["total_pii_spans"] += len(spans)

    for span in spans:
        pii_type = span.get("type")
        if pii_type in report["pii_counts_by_type"]:
            report["pii_counts_by_type"][pii_type] += 1


def build_global_summary(reports: List[Dict]) -> Dict:
    """
    Build the final global report.

    Args:
        reports: File-level reports.

    Returns:
        Global report dictionary.
    """
    total_files = len(reports)
    total_records = sum(report["total_records"] for report in reports)
    records_with_pii = sum(report["records_with_pii"] for report in reports)

    pii_counts_by_type = {
        "email": 0,
        "phone": 0,
        "ip": 0,
        "physical_address": 0,
    }

    for report in reports:
        for pii_type, count in report["pii_counts_by_type"].items():
            pii_counts_by_type[pii_type] += count

    return {
        "total_files": total_files,
        "total_records": total_records,
        "records_with_pii": records_with_pii,
        "records_with_pii_percentage": round(
            (records_with_pii / total_records) * 100,
            2,
        )
        if total_records
        else 0.0,
        "pii_counts_by_type": pii_counts_by_type,
    }


def save_report(report_path: Path, reports: List[Dict]) -> None:
    """
    Save only the global summary into one JSON file.

    Args:
        report_path: Output report path.
        reports: File-level reports.

    Returns:
        None.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output = build_global_summary(reports)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def init_datafog_clean() -> None:
    """
    Initialize DataFog while suppressing noisy output.

    Returns:
        None.
    """
    print("Inicializando DataFog...", end="", flush=True)

    start = time.time()
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        init_datafog()

    elapsed = time.time() - start
    print(f" OK ({elapsed:.2f}s)")


def process_file(path: Path, file_position: int, total_files: int) -> Dict:
    """
    Process one JSONL file and overwrite it.

    Args:
        path: Input JSONL file path.
        file_position: Current file index.
        total_files: Total number of files.

    Returns:
        File report dictionary.
    """
    records = list(read_jsonl(path))
    updated_records: List[Dict] = []
    report = init_file_report(path, len(records))

    desc = f"[{file_position}/{total_files}] {path.name}"

    pbar = tqdm(
        total=len(records),
        desc=desc,
        unit="rec",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for index, record in enumerate(records, start=1):
        text = extract_text(record)
        updated_record = compute_pii_metrics(record, text)

        updated_records.append(updated_record)
        update_file_report(report, updated_record)

        pbar.update(1)

        if index % 500 == 0 or index == len(records):
            pbar.set_postfix(
                processed=index,
                spans=report["total_pii_spans"],
                with_pii=report["records_with_pii"],
            )

    pbar.close()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp_path, updated_records)
    tmp_path.replace(path)

    return report


def main() -> None:
    """
    Run the PII processing pipeline over all JSONL files.

    Returns:
        None.
    """
    configure_runtime()
    init_datafog_clean()

    files = sorted(INPUT_DIR.glob("*.jsonl"))

    if not files:
        print(f"No JSONL files found in {INPUT_DIR}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports: List[Dict] = []

    files_bar = tqdm(
        files,
        desc="Processing files",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for index, path in enumerate(files_bar, start=1):
        files_bar.set_postfix(current=path.name)
        report = process_file(path, index, len(files))
        reports.append(report)

    save_report(REPORT_PATH, reports)

    print("\nProcessing completed.")
    print(f"Report saved in: {REPORT_PATH}")


if __name__ == "__main__":
    main()