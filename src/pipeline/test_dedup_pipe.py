from .url_deduplication import URLDeduplicator  # adjust import

if __name__ == "__main__":
    data_path = "data/processed"

    url_dedup_ignore = [
        "hplt-3",
        "orembae",
        "FinePDF",
        "fineweb-2",
        "moscar",
        "flores-200",
        "multi-wiki-qa"
    ]

    deduper = URLDeduplicator(
        file_directory=data_path,
        output_base_dir="outputs/url_deduplication",
        db_path="seen_urls.sqlite",
        exclude_files=url_dedup_ignore,
        reuse_db=True
    )

    deduper.run()