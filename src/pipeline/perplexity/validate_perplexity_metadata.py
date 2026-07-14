from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Dict, Iterator, List


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

OUTPUT_DIR = Path(
    os.getenv(
        "PERPLEXITY_REPORT_DIR",
        str(
            BASE_DIR
            / "outputs"
            / "report"
        ),
    )
)

LOG_PATH = Path(
    os.getenv(
        "PERPLEXITY_METADATA_LOG",
        str(
            OUTPUT_DIR
            / "perplexity_metadata.log"
        ),
    )
)

COREGUAPA_METRIC = "coreguapa_perplexity"
TWEETS_METRIC = "tweets_perplexity"


def read_jsonl(
    path: Path,
) -> Iterator[Dict]:
    """
    Read a JSONL file and yield one record per non-empty line.

    Args:
        path (Path): Path to the JSONL file.

    Yields:
        Dict: Parsed JSON record.

    Raises:
        ValueError: If a line does not contain a JSON object.
        json.JSONDecodeError: If a line contains invalid JSON.
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


def get_corpus_name(
    file_path: Path,
    record: Dict,
) -> str:
    """
    Resolve the corpus name for a document.

    The value of the root-level ``corpus`` field is used
    when available. Otherwise, the JSONL file stem is used.

    Args:
        file_path (Path): Source JSONL file.
        record (Dict): Document record.

    Returns:
        str: Corpus name.
    """
    corpus = record.get("corpus")

    if (
        isinstance(corpus, str)
        and corpus.strip()
    ):
        return corpus.strip()

    return file_path.stem


def record_has_metric(
    record: Dict,
    metric_name: str,
) -> bool:
    """
    Check whether a record contains a valid metric.

    Args:
        record (Dict): Document record.
        metric_name (str): Metric name.

    Returns:
        bool: True when the metric value is not None.
    """
    return record.get(metric_name) is not None


def initialize_corpus_report(
    corpus: str,
) -> Dict:
    """
    Create the counters for one corpus.

    Args:
        corpus (str): Corpus name.

    Returns:
        Dict: Initialized counters.
    """
    return {
        "corpus": corpus,
        "documents": 0,
        "documents_with_coreguapa_perplexity": 0,
        "documents_with_tweets_perplexity": 0,
    }


def update_report(
    reports_by_corpus: Dict[str, Dict],
    file_path: Path,
    record: Dict,
) -> None:
    """
    Update the counters for one document.

    Args:
        reports_by_corpus (Dict[str, Dict]):
            Reports indexed by corpus name.
        file_path (Path):
            Source JSONL file.
        record (Dict):
            Document record.
    """
    corpus = get_corpus_name(
        file_path=file_path,
        record=record,
    )

    if corpus not in reports_by_corpus:
        reports_by_corpus[corpus] = (
            initialize_corpus_report(corpus)
        )

    report = reports_by_corpus[corpus]

    report["documents"] += 1

    if record_has_metric(
        record,
        COREGUAPA_METRIC,
    ):
        report[
            "documents_with_coreguapa_perplexity"
        ] += 1

    if record_has_metric(
        record,
        TWEETS_METRIC,
    ):
        report[
            "documents_with_tweets_perplexity"
        ] += 1


def validate_reports(
    reports: List[Dict],
) -> bool:
    """
    Check whether every document contains both metrics.

    Args:
        reports (List[Dict]): Per-corpus reports.

    Returns:
        bool: True when all documents contain both metrics.
    """
    if not reports:
        return False

    for report in reports:
        total_documents = report["documents"]

        if total_documents <= 0:
            return False

        if (
            report[
                "documents_with_coreguapa_perplexity"
            ]
            != total_documents
        ):
            return False

        if (
            report[
                "documents_with_tweets_perplexity"
            ]
            != total_documents
        ):
            return False

    return True


def build_summary(
    reports: List[Dict],
) -> Dict:
    """
    Build global totals.

    Args:
        reports (List[Dict]): Per-corpus reports.

    Returns:
        Dict: Global summary.
    """
    return {
        "corpora_processed": len(reports),
        "total_documents": sum(
            report["documents"]
            for report in reports
        ),
        "documents_with_coreguapa_perplexity": sum(
            report[
                "documents_with_coreguapa_perplexity"
            ]
            for report in reports
        ),
        "documents_with_tweets_perplexity": sum(
            report[
                "documents_with_tweets_perplexity"
            ]
            for report in reports
        ),
    }


def write_report(
    reports: List[Dict],
    invalid_files: List[Dict],
    is_valid: bool,
) -> None:
    """
    Write the validation report as JSON.

    Args:
        reports (List[Dict]): Per-corpus reports.
        invalid_files (List[Dict]): Files that could not be read.
        is_valid (bool): Final validation status.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "status": (
            "valid"
            if is_valid
            else "invalid"
        ),
        "summary": build_summary(reports),
        "corpora": reports,
    }

    if invalid_files:
        output["errors"] = invalid_files

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")


def print_summary(
    reports: List[Dict],
    is_valid: bool,
) -> None:
    """
    Print a simplified validation table.

    Args:
        reports (List[Dict]): Per-corpus reports.
        is_valid (bool): Final validation status.
    """
    summary = build_summary(reports)

    corpus_width = max(
        30,
        max(
            (
                len(report["corpus"])
                for report in reports
            ),
            default=0,
        ),
    )

    separator_width = (
        corpus_width
        + 15
        + 18
        + 18
    )

    print()
    print("=" * separator_width)
    print("Perplexity metadata validation")
    print("=" * separator_width)

    print(
        f"{'Corpus':<{corpus_width}}"
        f"{'Documents':>15}"
        f"{'CoreGuapa':>18}"
        f"{'GN Tweets':>18}"
    )

    print("-" * separator_width)

    for report in reports:
        print(
            f"{report['corpus']:<{corpus_width}}"
            f"{report['documents']:>15}"
            f"{report[
                'documents_with_coreguapa_perplexity'
            ]:>18}"
            f"{report[
                'documents_with_tweets_perplexity'
            ]:>18}"
        )

    print("-" * separator_width)

    print(
        f"{'TOTAL':<{corpus_width}}"
        f"{summary['total_documents']:>15}"
        f"{summary[
            'documents_with_coreguapa_perplexity'
        ]:>18}"
        f"{summary[
            'documents_with_tweets_perplexity'
        ]:>18}"
    )

    print("=" * separator_width)

    print(
        f"Corpora processed : "
        f"{summary['corpora_processed']}"
    )
    print(
        f"Total documents   : "
        f"{summary['total_documents']}"
    )
    print(
        f"With CoreGuapa    : "
        f"{summary[
            'documents_with_coreguapa_perplexity'
        ]}"
    )
    print(
        f"With GN Tweets    : "
        f"{summary[
            'documents_with_tweets_perplexity'
        ]}"
    )
    print(
        f"Validation status : "
        f"{'PASS' if is_valid else 'FAIL'}"
    )
    print(
        f"Report saved to   : "
        f"{LOG_PATH}"
    )
    print("=" * separator_width)


def main() -> None:
    """
    Validate perplexity metadata in every JSONL document.

    The validation passes only when every document contains
    both ``coreguapa_perplexity`` and ``tweets_perplexity``.
    """
    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:
        write_report(
            reports=[],
            invalid_files=[],
            is_valid=False,
        )

        raise SystemExit(
            f"No JSONL files found in {INPUT_DIR}."
        )

    reports_by_corpus: Dict[str, Dict] = {}
    invalid_files: List[Dict] = []

    for file_path in files:
        try:
            for record in read_jsonl(file_path):
                update_report(
                    reports_by_corpus=reports_by_corpus,
                    file_path=file_path,
                    record=record,
                )

        except (
            json.JSONDecodeError,
            ValueError,
            OSError,
        ) as error:
            invalid_files.append(
                {
                    "file": str(file_path),
                    "error": str(error),
                }
            )

    reports = sorted(
        reports_by_corpus.values(),
        key=lambda report: report["corpus"],
    )

    is_valid = (
        validate_reports(reports)
        and not invalid_files
    )

    write_report(
        reports=reports,
        invalid_files=invalid_files,
        is_valid=is_valid,
    )

    print_summary(
        reports=reports,
        is_valid=is_valid,
    )

    if invalid_files:
        print()
        print("Files that could not be validated:")

        for invalid_file in invalid_files:
            print(
                f"- {invalid_file['file']}: "
                f"{invalid_file['error']}"
            )

    if not is_valid:
        raise SystemExit(
            "Some documents are missing perplexity "
            "metadata or could not be validated."
        )


if __name__ == "__main__":
    main()