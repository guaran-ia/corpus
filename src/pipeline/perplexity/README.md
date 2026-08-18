# Perplexity Computation Pipeline

This module computes *perplexity* metrics for JSONL documents using the
**CoreGuapa LM** and **GN Tweets LM** models.

The computed metrics are stored directly at the root of each record:

``` json
{
  "text": "...",
  "coreguapa_perplexity": 18.42,
  "tweets_perplexity": 131.77
}
```

------------------------------------------------------------------------

# Models

-   CoreGuapa LM: https://huggingface.co/guaran-ia/coreguapa-lm
-   GN Tweets LM: https://huggingface.co/guaran-ia/gntweets-lm

------------------------------------------------------------------------

# Project structure

``` text
src/pipeline/perplexity
```

------------------------------------------------------------------------

# Environment variables

``` bash
PERPLEXITY_INPUT_DIR=/workspace/corpus/data/processed

HF_HOME=/workspace/corpus/.cache/huggingface
HF_HUB_CACHE=/workspace/corpus/.cache/huggingface/hub
HF_LOCAL_FILES_ONLY=0

PERPLEXITY_MAX_LENGTH=8192
PERPLEXITY_STRIDE=4096
PERPLEXITY_TEXT_CHUNK_SIZE=32768

PERPLEXITY_DOCUMENT_BATCH_SIZE=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

  -----------------------------------------------------------------------
  Variable                       Description
  ------------------------------ ----------------------------------------
  `PERPLEXITY_INPUT_DIR`         Directory containing the processed JSONL
                                 files.

  `PERPLEXITY_MAX_LENGTH`        Maximum number of tokens processed per
                                 inference.

  `PERPLEXITY_STRIDE`            Tokens preserved as context between
                                 consecutive windows.

  `PERPLEXITY_TEXT_CHUNK_SIZE`   Size of the text chunk used during
                                 incremental tokenization.

  `PERPLEXITY_DOCUMENT_BATCH_SIZE`
                                 Number of documents processed per batch.

  `HF_HOME`, `HF_HUB_CACHE`      Hugging Face cache directories.

  `HF_LOCAL_FILES_ONLY`          Uses only locally stored models when
                                 enabled.

  `PYTORCH_CUDA_ALLOC_CONF`      PyTorch CUDA memory allocator
                                 configuration.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Pipeline workflow

1.  Read the JSONL documents.
2.  Extract the text from each document.
3.  Split the text into chunks to reduce memory usage during
    tokenization.
4.  Tokenize each chunk incrementally.
5.  Generate sliding windows using `PERPLEXITY_MAX_LENGTH` and
    `PERPLEXITY_STRIDE`.
6.  Group the windows for inference.
7.  Convert the windows into PyTorch tensors.
8.  Apply *padding* to shorter windows.
9.  Run the model.
10. Compute the loss for each window.
11. Accumulate the loss for each document.
12. Compute the final *perplexity*.
13. Store the metric in the JSONL record.

------------------------------------------------------------------------

# Memory management

The implementation avoids tokenizing the entire document in a single
operation.

Instead, it:

-   tokenizes one text chunk at a time;
-   keeps only the tokens required to build the windows;
-   processes only the windows needed for each inference.

This approach allows processing documents whose length greatly exceeds
the model's maximum context size.

------------------------------------------------------------------------

# Sliding windows

When a document exceeds `PERPLEXITY_MAX_LENGTH`, the pipeline generates
overlapping windows.

The context between consecutive windows is preserved using
`PERPLEXITY_STRIDE`.

The implementation follows the strategy described in the model cards.

------------------------------------------------------------------------

# Inference preparation

The windows are converted into PyTorch tensors before executing the
model.

Shorter windows are padded to build rectangular tensors.

Padding tokens do not participate in the model attention mechanism or in
the loss computation.

------------------------------------------------------------------------

# Perplexity computation

Each window is evaluated independently.

The accumulated loss from all windows belonging to a document is used to
compute the final *perplexity*.

``` text
average_negative_log_likelihood =
    total_negative_log_likelihood /
    total_window_tokens

perplexity =
    exp(average_negative_log_likelihood)
```

------------------------------------------------------------------------

# Execution

## Compute only CoreGuapa

``` bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model coreguapa
```

Generates only the `coreguapa_perplexity` metric.

## Compute only GN Tweets

``` bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model tweets
```

Generates only the `tweets_perplexity` metric.

## Compute both metrics

``` bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model all
```

Loads both models and computes the missing metrics for each document.

## Recommendation

To ensure that the corpus contains both metrics, you must either:

-   run `--model coreguapa` followed by `--model tweets`, or
-   run `--model all` directly.

The `--model all` option is the recommended way to compute both metrics
in a single execution.

------------------------------------------------------------------------

# Resuming executions

Existing metrics are preserved.

Only missing metrics are computed, allowing interrupted executions to
continue without repeating previously completed work.

------------------------------------------------------------------------

# Validation

Once the corpus contains both metrics, run:

``` bash
python -m src.pipeline.perplexity.validate_perplexity_metadata
```

The report is generated at:

``` text
outputs/report/perplexity_metadata.log
```

> **Important**
>
> The validation process checks the final state of the corpus and
> expects **every document** to contain both `coreguapa_perplexity` and
> `tweets_perplexity`.
>
> If only `--model coreguapa` or `--model tweets` has been executed, the
> validation will report missing metrics and finish with **FAIL**. This
> is the expected behavior.
>
> To obtain a successful validation, both metrics must have been
> computed, either by running the two models separately or by using
> `--model all`.

------------------------------------------------------------------------

# Output

Each JSONL record contains the computed metrics at the root of the
document.

``` json
{
  "text": "...",
  "coreguapa_perplexity": 18.42,
  "tweets_perplexity": 131.77
}
```