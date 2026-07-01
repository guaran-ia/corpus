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


def read_jsonl(
    path: Path,
) -> Iterator[Dict]:
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


def get_corpus_name(
    file_path: Path,
    record: Dict,
) -> str:
    """
    Resolve the corpus name for a document record.

    Args:
        file_path (Path): Path to the source JSONL file.
        record (Dict): Corpus document record.

    Returns:
        str: Corpus name from the record when available,
            otherwise the file stem.
    """
    corpus = record.get(
        "corpus"
    )

    if (
        isinstance(corpus, str)
        and corpus.strip()
    ):
        return corpus

    return file_path.stem


def update_report(
    reports_by_corpus: Dict[str, Dict],
    file_path: Path,
    record: Dict,
) -> None:
    """
    Update validation counters for one corpus document.

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
        reports_by_corpus[corpus] = {
            "corpus": corpus,
            "total_records": 0,
            "records_with_coreguapa_perplexity": 0,
            "records_with_tweets_perplexity": 0,
        }

    report = reports_by_corpus[
        corpus
    ]

    report["total_records"] += 1

    metadata = record.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    if (
        metadata.get(
            "coreguapa_perplexity"
        )
        is not None
    ):
        report[
            "records_with_coreguapa_perplexity"
        ] += 1

    if (
        metadata.get(
            "tweets_perplexity"
        )
        is not None
    ):
        report[
            "records_with_tweets_perplexity"
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
    is_valid = True

    for report in reports:

        total = report[
            "total_records"
        ]

        if (
            report[
                "records_with_coreguapa_perplexity"
            ]
            != total
        ):
            is_valid = False

        if (
            report[
                "records_with_tweets_perplexity"
            ]
            != total
        ):
            is_valid = False

    return is_valid


def main() -> None:
    """
    Validate perplexity metadata for all processed corpora
    and save a validation report.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit:
            If any document is missing at least one
            required perplexity metric.
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

    for file_path in files:

        for record in read_jsonl(
            file_path
        ):
            update_report(
                reports_by_corpus,
                file_path,
                record,
            )

    reports = sorted(
        reports_by_corpus.values(),
        key=lambda item: item[
            "corpus"
        ],
    )

    is_valid = validate_reports(
        reports
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Validation saved to {LOG_PATH}"
    )

    if not is_valid:
        raise SystemExit(
            "Some documents are missing perplexity metadata."
        )


if __name__ == "__main__":
    main()

