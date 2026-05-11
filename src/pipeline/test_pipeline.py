from .pipeline import Pipeline
from .deduplication.remove_duplicates_url import URLDeduplication
from .deduplication.remove_duplicates_minhash import MinHashDeduplication

if __name__ == "__main__":
    input_directory = "data/processed"
    output_directory = "outputs/pipeline_test"

    url_dedup = URLDeduplication(
        duplicate_ids_path="outputs/deduplication/url_202605022046/duplicate_ids.json",
        ignore_files_path="outputs/deduplication/url_dedup_ignore_corpora.json"
    )

    minhash_dedup = MinHashDeduplication(
        duplicate_ids_path="outputs/deduplication/minhash_202602201929/duplicates.json"
    )

    pipeline = Pipeline(steps = [url_dedup, minhash_dedup], 
                        input_directory=input_directory, 
                        output_directory=output_directory
                )

    pipeline.run()