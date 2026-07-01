# Perplexity Metadata Pipeline

This module computes perplexity metrics for the corpus documents and stores the results in the `metadata` field of each JSONL record.

The following metrics are generated:

- `coreguapa_perplexity`: computed using the `guaran-ia/coreguapa-lm` model.
- `tweets_perplexity`: computed using the `guaran-ia/gntweets-lm` model.

## Location

```bash
src/pipeline/perplexity
```

---

# Configuration

Before running the pipeline, configure the following environment variables.

The paths are relative to the repository root.

```bash
export PERPLEXITY_INPUT_DIR=data/processed

export HF_HOME=.cache/huggingface
export HF_HUB_CACHE=.cache/huggingface/hub
export HF_LOCAL_FILES_ONLY=0

export PERPLEXITY_MAX_LENGTH=8192
export PERPLEXITY_STRIDE=4096

export BATCH_SIZE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PERPLEXITY_INPUT_DIR` | Directory containing the JSONL files to process. |
| `HF_HOME` | Directory where Hugging Face stores downloaded models. |
| `HF_HUB_CACHE` | Hugging Face Hub cache directory. |
| `HF_LOCAL_FILES_ONLY` | If set to `1`, only locally cached models are used. |
| `PERPLEXITY_MAX_LENGTH` | Maximum number of tokens processed in each sliding window. |
| `PERPLEXITY_STRIDE` | Number of tokens the sliding window advances between consecutive windows. |
| `BATCH_SIZE` | Number of documents processed simultaneously. |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memory allocation configuration used to reduce memory fragmentation. |

---

# Hardware

The experiments and metric generation were performed using the following GPU:

- **NVIDIA RTX A6000**
- **48 GB VRAM**
- **CUDA 12.8**

The available memory allows the models to be loaded entirely and enables the processing of long documents using sliding windows without loading both models into memory simultaneously.

---

# Long Document Processing

The models have a maximum context length. To avoid losing information when processing documents longer than this limit, the pipeline implements a **Sliding Window** strategy.

With the following configuration:

```bash
PERPLEXITY_MAX_LENGTH=8192
PERPLEXITY_STRIDE=4096
```

each document is divided into windows of up to **8192 tokens**.

Each new window starts **4096 tokens after the previous one**, creating an overlap between consecutive windows.

This approach allows the pipeline to:

- process documents of arbitrary length;
- avoid truncating the input text;
- preserve the model's contextual information;
- compute perplexity using the entire document.

---

# Running the Pipeline

## Compute CoreGuapa Perplexity Only

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model coreguapa
```

---

## Compute GN Tweets Perplexity Only

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model tweets
```

---

## Compute Both Metrics

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model all
```

When using the `all` option, the pipeline:

1. loads only the **CoreGuapa** model;
2. processes every corpus file;
3. releases the allocated memory;
4. loads the **GN Tweets** model;
5. processes every corpus file again.

This ensures that both models are never loaded into memory at the same time.

---

# Resuming Interrupted Executions

Before computing a metric, the pipeline checks whether the corresponding metadata already exists in the document.

For example:

```json
{
    "metadata": {
        "coreguapa_perplexity": 36.15
    }
}
```

If the requested metric is already present, that document is skipped.

This behavior allows the pipeline to:

- resume interrupted executions;
- avoid recomputing existing metrics;
- compute only the missing metadata.

---

# Validation

After the computation finishes, the generated metadata can be validated by running:

```bash
python -m src.pipeline.perplexity.validate_perplexity_metadata
```

The validation report is written to:

```text
outputs/report/perplexity_metadata.log
```

The report contains, for each corpus:

- total number of documents;
- number of documents containing `coreguapa_perplexity`;
- number of documents containing `tweets_perplexity`.

Example:

```json
[
    {
        "corpus": "udhr-lid",
        "total_records": 122,
        "records_with_coreguapa_perplexity": 122,
        "records_with_tweets_perplexity": 122
    }
]
```