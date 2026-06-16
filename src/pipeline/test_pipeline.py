from .pipeline import Pipeline
from .pipeline_steps.url_deduplication import URLDeduplication
from .pipeline_steps.minhash_deduplication import MinHashDeduplication

if __name__ == "__main__":
    input_directory = "data/processed"
    output_directory = "outputs/pipeline_test"

    url_dedup = URLDeduplication(
        duplicate_ids_path="outputs/deduplication/url_202606011252/duplicate_ids.json",
        ignore_files_path="outputs/deduplication/url_dedup_ignore_corpora.json"
    )

    minhash_dedup = MinHashDeduplication(
        duplicate_ids_path="outputs/deduplication/minhash_202606111735/duplicates.json"
    )

    pipeline = Pipeline(steps = [url_dedup, minhash_dedup], 
                        input_directory=input_directory, 
                        output_directory=output_directory
                )

    pipeline.run()