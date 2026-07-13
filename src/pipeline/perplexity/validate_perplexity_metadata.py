from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

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

REQUIRED_METRICS = (
    "coreguapa_perplexity",
    "tweets_perplexity",
)


def read_jsonl(
    path: Path,
) -> Iterator[Tuple[int, Dict]]:
    """
    Read a JSONL file and yield one record per non-empty line.

    Args:
        path (Path):
            Path to the JSONL file.

    Yields:
        Tuple[int, Dict]:
            Line number and parsed JSON record.

    Raises:
        ValueError:
            If a JSONL line does not contain a JSON object.
        json.JSONDecodeError:
            If a non-empty line contains invalid JSON.
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


def get_corpus_name(
    file_path: Path,
    record: Dict,
) -> str:
    """
    Resolve the corpus name for a document record.

    Args:
        file_path (Path):
            Path to the source JSONL file.
        record (Dict):
            Corpus document record.

    Returns:
        str:
            Corpus name from the record when available,
            otherwise the file stem.
    """
    corpus = record.get(
        "corpus"
    )

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
    Check whether a document contains a valid perplexity metric.

    Perplexity metrics are expected directly at the root level
    of each document record.

    Args:
        record (Dict):
            Corpus document record.
        metric_name (str):
            Perplexity metric name.

    Returns:
        bool:
            True when the metric exists and its value is not None.
    """
    return (
        metric_name in record
        and record.get(
            metric_name
        )
        is not None
    )


def initialize_corpus_report(
    corpus: str,
) -> Dict:
    """
    Create an empty validation report for one corpus.

    Args:
        corpus (str):
            Corpus name.

    Returns:
        Dict:
            Initialized report counters.
    """
    return {
        "corpus": corpus,
        "total_records": 0,
        "records_with_coreguapa_perplexity": 0,
        "records_with_tweets_perplexity": 0,
        "records_with_both_perplexities": 0,
        "records_missing_coreguapa_perplexity": 0,
        "records_missing_tweets_perplexity": 0,
        "records_missing_any_perplexity": 0,
        "source_files": [],
    }


def update_report(
    reports_by_corpus: Dict[str, Dict],
    file_path: Path,
    record: Dict,
) -> None:
    """
    Update validation counters for one corpus document.

    Perplexity values are read directly from the document root,
    for example:

        {
            "text": "...",
            "coreguapa_perplexity": 12.45,
            "tweets_perplexity": 34.67
        }

    Args:
        reports_by_corpus (Dict[str, Dict]):
            Reports indexed by corpus name.
        file_path (Path):
            Path to the source JSONL file.
        record (Dict):
            Corpus document record.

    Returns:
        None.
    """
    corpus = get_corpus_name(
        file_path,
        record,
    )

    if corpus not in reports_by_corpus:
        reports_by_corpus[
            corpus
        ] = initialize_corpus_report(
            corpus
        )

    report = reports_by_corpus[
        corpus
    ]

    file_name = file_path.name

    if file_name not in report[
        "source_files"
    ]:
        report[
            "source_files"
        ].append(
            file_name
        )

    has_coreguapa = record_has_metric(
        record,
        "coreguapa_perplexity",
    )

    has_tweets = record_has_metric(
        record,
        "tweets_perplexity",
    )

    report[
        "total_records"
    ] += 1

    if has_coreguapa:
        report[
            "records_with_coreguapa_perplexity"
        ] += 1

    else:
        report[
            "records_missing_coreguapa_perplexity"
        ] += 1

    if has_tweets:
        report[
            "records_with_tweets_perplexity"
        ] += 1

    else:
        report[
            "records_missing_tweets_perplexity"
        ] += 1

    if (
        has_coreguapa
        and has_tweets
    ):
        report[
            "records_with_both_perplexities"
        ] += 1

    else:
        report[
            "records_missing_any_perplexity"
        ] += 1


def validate_reports(
    reports: List[Dict],
) -> bool:
    """
    Check whether every corpus report has complete perplexity metadata.

    Args:
        reports (List[Dict]):
            Per-corpus validation reports.

    Returns:
        bool:
            True when every document has both
            perplexity metrics.
    """
    if not reports:
        return False

    for report in reports:

        total = report[
            "total_records"
        ]

        if total <= 0:
            return False

        if (
            report[
                "records_with_coreguapa_perplexity"
            ]
            != total
        ):
            return False

        if (
            report[
                "records_with_tweets_perplexity"
            ]
            != total
        ):
            return False

        if (
            report[
                "records_with_both_perplexities"
            ]
            != total
        ):
            return False

    return True


def build_summary(
    reports: List[Dict],
    files: List[Path],
    invalid_records: List[Dict],
) -> Dict:
    """
    Build the global validation summary.

    Args:
        reports (List[Dict]):
            Per-corpus validation reports.
        files (List[Path]):
            JSONL files discovered in the input directory.
        invalid_records (List[Dict]):
            Records describing files that could not be validated.

    Returns:
        Dict:
            Global validation totals.
    """
    total_records = sum(
        report[
            "total_records"
        ]
        for report in reports
    )

    records_with_coreguapa = sum(
        report[
            "records_with_coreguapa_perplexity"
        ]
        for report in reports
    )

    records_with_tweets = sum(
        report[
            "records_with_tweets_perplexity"
        ]
        for report in reports
    )

    records_with_both = sum(
        report[
            "records_with_both_perplexities"
        ]
        for report in reports
    )

    return {
        "input_directory": str(
            INPUT_DIR
        ),
        "files_found": len(
            files
        ),
        "corpora_found": len(
            reports
        ),
        "total_records": total_records,
        "records_with_coreguapa_perplexity": (
            records_with_coreguapa
        ),
        "records_with_tweets_perplexity": (
            records_with_tweets
        ),
        "records_with_both_perplexities": (
            records_with_both
        ),
        "records_missing_coreguapa_perplexity": (
            total_records
            - records_with_coreguapa
        ),
        "records_missing_tweets_perplexity": (
            total_records
            - records_with_tweets
        ),
        "records_missing_any_perplexity": (
            total_records
            - records_with_both
        ),
        "invalid_files": len(
            invalid_records
        ),
    }


def write_report(
    reports: List[Dict],
    files: List[Path],
    invalid_records: List[Dict],
    is_valid: bool,
) -> None:
    """
    Save the perplexity metadata validation report.

    Args:
        reports (List[Dict]):
            Per-corpus validation reports.
        files (List[Path]):
            JSONL files discovered in the input directory.
        invalid_records (List[Dict]):
            Files or lines that could not be validated.
        is_valid (bool):
            Final validation status.

    Returns:
        None.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = build_summary(
        reports=reports,
        files=files,
        invalid_records=invalid_records,
    )

    output = {
        "status": (
            "valid"
            if is_valid
            else "invalid"
        ),
        "required_metrics": list(
            REQUIRED_METRICS
        ),
        "summary": summary,
        "corpora": reports,
        "errors": invalid_records,
    }

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

        file.write(
            "\n"
        )


def print_summary(
    reports: List[Dict],
    files: List[Path],
    invalid_records: List[Dict],
    is_valid: bool,
) -> None:
    """
    Print a compact validation summary.

    Args:
        reports (List[Dict]):
            Per-corpus validation reports.
        files (List[Path]):
            JSONL files discovered in the input directory.
        invalid_records (List[Dict]):
            Files or records that could not be validated.
        is_valid (bool):
            Final validation status.

    Returns:
        None.
    """
    summary = build_summary(
        reports=reports,
        files=files,
        invalid_records=invalid_records,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "Perplexity metadata validation"
    )
    print(
        "=" * 80
    )
    print(
        f"Input directory                  : "
        f"{INPUT_DIR}"
    )
    print(
        f"JSONL files found               : "
        f"{summary['files_found']}"
    )
    print(
        f"Corpora found                    : "
        f"{summary['corpora_found']}"
    )
    print(
        f"Total documents                  : "
        f"{summary['total_records']}"
    )
    print(
        f"With coreguapa_perplexity        : "
        f"{summary['records_with_coreguapa_perplexity']}"
    )
    print(
        f"With tweets_perplexity           : "
        f"{summary['records_with_tweets_perplexity']}"
    )
    print(
        f"With both metrics                : "
        f"{summary['records_with_both_perplexities']}"
    )
    print(
        f"Missing coreguapa_perplexity     : "
        f"{summary['records_missing_coreguapa_perplexity']}"
    )
    print(
        f"Missing tweets_perplexity        : "
        f"{summary['records_missing_tweets_perplexity']}"
    )
    print(
        f"Missing at least one metric      : "
        f"{summary['records_missing_any_perplexity']}"
    )
    print(
        f"Invalid files                    : "
        f"{summary['invalid_files']}"
    )
    print(
        f"Validation status                : "
        f"{'PASS' if is_valid else 'FAIL'}"
    )
    print(
        f"Report saved to                  : "
        f"{LOG_PATH}"
    )
    print(
        "=" * 80
    )


def main() -> None:
    """
    Validate perplexity metadata for all processed corpora
    and save a validation report.

    Perplexity metrics are expected at the root level of each
    JSONL document.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit:
            If no JSONL files are found, a file cannot be read,
            or any document is missing at least one required
            perplexity metric.
    """
    files = sorted(
        INPUT_DIR.glob(
            "*.jsonl"
        )
    )

    reports_by_corpus: Dict[
        str,
        Dict,
    ] = {}

    invalid_records: List[
        Dict
    ] = []

    if not files:
        write_report(
            reports=[],
            files=[],
            invalid_records=[],
            is_valid=False,
        )

        raise SystemExit(
            f"No JSONL files found in {INPUT_DIR}."
        )

    for file_path in files:

        try:
            for (
                line_number,
                record,
            ) in read_jsonl(
                file_path
            ):
                update_report(
                    reports_by_corpus,
                    file_path,
                    record,
                )

        except (
            json.JSONDecodeError,
            ValueError,
            OSError,
        ) as error:
            invalid_records.append(
                {
                    "file": str(
                        file_path
                    ),
                    "error": str(
                        error
                    ),
                }
            )

    reports = sorted(
        reports_by_corpus.values(),
        key=lambda item: item[
            "corpus"
        ],
    )

    for report in reports:
        report[
            "source_files"
        ] = sorted(
            report[
                "source_files"
            ]
        )

    is_valid = (
        validate_reports(
            reports
        )
        and not invalid_records
    )

    write_report(
        reports=reports,
        files=files,
        invalid_records=invalid_records,
        is_valid=is_valid,
    )

    print_summary(
        reports=reports,
        files=files,
        invalid_records=invalid_records,
        is_valid=is_valid,
    )

    if not is_valid:
        raise SystemExit(
            "Some documents are missing perplexity metadata "
            "or could not be validated."
        )


if __name__ == "__main__":
    main()
