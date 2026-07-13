from __future__ import annotations

import gc
import math
import os
import threading

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional

from transformers import AutoModelForCausalLM, AutoTokenizer


COREGUAPA_MODEL_ID = "guaran-ia/coreguapa-lm"
GNTWEETS_MODEL_ID = "guaran-ia/gntweets-lm"

MAX_LENGTH = int(
    os.getenv(
        "PERPLEXITY_MAX_LENGTH",
        "8192",
    )
)

STRIDE = int(
    os.getenv(
        "PERPLEXITY_STRIDE",
        "4096",
    )
)

TEXT_CHUNK_SIZE = int(
    os.getenv(
        "PERPLEXITY_TEXT_CHUNK_SIZE",
        "32768",
    )
)

CONFIGURED_BATCH_SIZE = int(
    os.getenv(
        "PERPLEXITY_WINDOW_BATCH_SIZE",
        "0",
    )
)

MIN_WINDOW_BATCH_SIZE = int(
    os.getenv(
        "PERPLEXITY_MIN_WINDOW_BATCH_SIZE",
        "1",
    )
)

MAX_WINDOW_BATCH_SIZE = int(
    os.getenv(
        "PERPLEXITY_MAX_WINDOW_BATCH_SIZE",
        "64",
    )
)

GPU_MEMORY_UTILIZATION = float(
    os.getenv(
        "PERPLEXITY_GPU_MEMORY_UTILIZATION",
        "0.88",
    )
)

ESTIMATED_BYTES_PER_TOKEN = int(
    os.getenv(
        "PERPLEXITY_ESTIMATED_BYTES_PER_TOKEN",
        str(1024 * 1024),
    )
)

HF_HOME = os.getenv(
    "HF_HOME",
    "/disk/corpus/.cache/huggingface",
)

CACHE_DIR = os.getenv(
    "HF_HUB_CACHE",
    f"{HF_HOME}/hub",
)

LOCAL_FILES_ONLY = (
    os.getenv(
        "HF_LOCAL_FILES_ONLY",
        "0",
    )
    == "1"
)

MODEL_BY_METRIC = {
    "coreguapa_perplexity": COREGUAPA_MODEL_ID,
    "tweets_perplexity": GNTWEETS_MODEL_ID,
}


@dataclass
class LoadedModel:
    """
    Store the resources associated with one loaded language model.

    Attributes:
        metric_name (str): Metadata metric computed by the model.
        model_id (str): Hugging Face model identifier.
        tokenizer: Tokenizer associated with the model.
        model: Loaded causal language model.
        device (torch.device): Device receiving model inputs.
        stream: Optional CUDA stream used by this model.
        lock (threading.Lock): Lock preventing concurrent use of one model.
    """

    metric_name: str
    model_id: str
    tokenizer: object
    model: object
    device: torch.device
    stream: Optional[torch.cuda.Stream]
    lock: threading.Lock


@dataclass
class WindowTask:
    """
    Represent one sliding-window inference task.

    Each task corresponds to one overlapping chunk generated according to
    the recipe published in the model cards. The complete window contributes
    to the window loss, including tokens shared with adjacent windows.

    Attributes:
        document_index (int): Position of the source document in the batch.
        token_ids (List[int]): Token IDs in the current model window.
    """

    document_index: int
    token_ids: List[int]


_MODEL_REGISTRY: Dict[str, LoadedModel] = {}
_MODEL_REGISTRY_LOCK = threading.Lock()


def get_available_cuda_devices() -> List[int]:
    """
    Return the indexes of available CUDA devices.

    Returns:
        List[int]: CUDA device indexes.
    """
    if not torch.cuda.is_available():
        return []

    return list(
        range(torch.cuda.device_count())
    )


def get_device_memory(device_index: int) -> Tuple[int, int]:
    """
    Return free and total memory for one CUDA device.

    Args:
        device_index (int): CUDA device index.

    Returns:
        Tuple[int, int]: Free and total memory in bytes.
    """
    with torch.cuda.device(device_index):
        free_memory, total_memory = torch.cuda.mem_get_info()

    return int(free_memory), int(total_memory)


def select_device_for_metric(
    metric_name: str,
    metric_position: int,
) -> torch.device:
    """
    Select the preferred device for one metric.

    Models are distributed across CUDA devices when multiple GPUs are
    available. When only one GPU is available, all models share that GPU.

    Args:
        metric_name (str): Metadata metric name.
        metric_position (int): Position of the metric in the load order.

    Returns:
        torch.device: Selected device.
    """
    del metric_name

    cuda_devices = get_available_cuda_devices()

    if not cuda_devices:
        return torch.device("cpu")

    device_index = cuda_devices[
        metric_position % len(cuda_devices)
    ]

    return torch.device(
        f"cuda:{device_index}"
    )


def unload_model(metric_name: Optional[str] = None) -> None:
    """
    Unload one model or every model from the registry.

    Args:
        metric_name (Optional[str]): Metric whose model should be unloaded.
            When omitted, every loaded model is removed.
    """
    with _MODEL_REGISTRY_LOCK:
        if metric_name is None:
            metric_names = list(
                _MODEL_REGISTRY.keys()
            )
        else:
            metric_names = [metric_name]

        for current_metric_name in metric_names:
            loaded_model = _MODEL_REGISTRY.pop(
                current_metric_name,
                None,
            )

            if loaded_model is None:
                continue

            loaded_model.model = None
            loaded_model.tokenizer = None
            loaded_model.stream = None

    gc.collect()

    if torch.cuda.is_available():
        for device_index in get_available_cuda_devices():
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def load_model(
    metric_name: str,
    model_id: str,
    device: torch.device,
) -> LoadedModel:
    """
    Load a tokenizer and causal language model.

    Args:
        metric_name (str): Metadata metric computed by the model.
        model_id (str): Hugging Face model identifier.
        device (torch.device): Device where the model should be loaded.

    Returns:
        LoadedModel: Loaded model resources.

    Raises:
        RuntimeError: If loading fails.
    """
    print(
        f"Loading model {model_id} on {device} "
        f"using cache_dir={CACHE_DIR}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={},
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
            "cache_dir": CACHE_DIR,
            "local_files_only": LOCAL_FILES_ONLY,
        }

        if device.type == "cuda":
            model_kwargs["device_map"] = {
                "": str(device)
            }
        else:
            model_kwargs["device_map"] = None

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        if device.type == "cpu":
            model.to(device)

        model.eval()

        stream = None

        if device.type == "cuda":
            stream = torch.cuda.Stream(
                device=device
            )

        return LoadedModel(
            metric_name=metric_name,
            model_id=model_id,
            tokenizer=tokenizer,
            model=model,
            device=get_model_input_device(
                model=model,
                fallback_device=device,
            ),
            stream=stream,
            lock=threading.Lock(),
        )

    except Exception as error:
        unload_model(metric_name)

        raise RuntimeError(
            f"Failed to load model {model_id} "
            f"on device={device} "
            f"from cache_dir={CACHE_DIR}. "
            f"local_files_only={LOCAL_FILES_ONLY}. "
            f"Original error: {error}"
        ) from error


def load_models(metric_names: Sequence[str]) -> None:
    """
    Load all requested models and keep them available simultaneously.

    Args:
        metric_names (Sequence[str]): Metrics whose models must be loaded.

    Raises:
        ValueError: If a metric is unsupported.
        RuntimeError: If a model cannot be loaded.
    """
    unique_metric_names = list(
        dict.fromkeys(metric_names)
    )

    for metric_name in unique_metric_names:
        if metric_name not in MODEL_BY_METRIC:
            raise ValueError(
                f"Unsupported metric name: {metric_name}"
            )

    for metric_position, metric_name in enumerate(
        unique_metric_names
    ):
        with _MODEL_REGISTRY_LOCK:
            if metric_name in _MODEL_REGISTRY:
                continue

        device = select_device_for_metric(
            metric_name=metric_name,
            metric_position=metric_position,
        )

        loaded_model = load_model(
            metric_name=metric_name,
            model_id=MODEL_BY_METRIC[metric_name],
            device=device,
        )

        with _MODEL_REGISTRY_LOCK:
            _MODEL_REGISTRY[metric_name] = loaded_model


def get_model_for_metric(metric_name: str) -> LoadedModel:
    """
    Return the loaded resources associated with a metric.

    Args:
        metric_name (str): Metadata metric to compute.

    Returns:
        LoadedModel: Loaded model resources.

    Raises:
        ValueError: If the metric is unsupported.
    """
    if metric_name not in MODEL_BY_METRIC:
        raise ValueError(
            f"Unsupported metric name: {metric_name}"
        )

    with _MODEL_REGISTRY_LOCK:
        loaded_model = _MODEL_REGISTRY.get(
            metric_name
        )

    if loaded_model is None:
        load_models([metric_name])

        with _MODEL_REGISTRY_LOCK:
            loaded_model = _MODEL_REGISTRY[
                metric_name
            ]

    return loaded_model


def get_effective_max_length(
    tokenizer,
    model,
) -> int:
    """
    Resolve the effective model context length.

    The model configuration is preferred over tokenizer.model_max_length,
    because some tokenizers may expose a stale or generic value.

    Args:
        tokenizer: Tokenizer associated with the model.
        model: Loaded causal language model.

    Returns:
        int: Effective maximum window length.
    """
    model_max_length = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        isinstance(model_max_length, int)
        and model_max_length > 1
    ):
        return min(
            model_max_length,
            MAX_LENGTH,
        )

    tokenizer_max_length = getattr(
        tokenizer,
        "model_max_length",
        MAX_LENGTH,
    )

    if (
        tokenizer_max_length is None
        or tokenizer_max_length > 1_000_000
    ):
        return MAX_LENGTH

    return min(
        int(tokenizer_max_length),
        MAX_LENGTH,
    )


def validate_window_config(
    max_length: int,
) -> None:
    """
    Validate sliding-window and tokenization configuration.

    Args:
        max_length (int): Effective model context length.

    Raises:
        ValueError: If a configuration value is invalid.
    """
    if max_length <= 1:
        raise ValueError(
            "PERPLEXITY_MAX_LENGTH must be greater than 1. "
            f"Current effective value: {max_length}"
        )

    if STRIDE <= 0:
        raise ValueError(
            "PERPLEXITY_STRIDE must be greater than 0. "
            f"Current value: {STRIDE}"
        )

    if STRIDE >= max_length:
        raise ValueError(
            "PERPLEXITY_STRIDE must be smaller than the "
            "effective maximum context length so consecutive "
            "windows overlap. "
            f"Current stride: {STRIDE}. "
            f"Effective max length: {max_length}."
        )

    if TEXT_CHUNK_SIZE <= 0:
        raise ValueError(
            "PERPLEXITY_TEXT_CHUNK_SIZE must be greater than 0. "
            f"Current value: {TEXT_CHUNK_SIZE}"
        )

    if MIN_WINDOW_BATCH_SIZE <= 0:
        raise ValueError(
            "PERPLEXITY_MIN_WINDOW_BATCH_SIZE must be greater "
            f"than 0. Current value: {MIN_WINDOW_BATCH_SIZE}"
        )

    if MAX_WINDOW_BATCH_SIZE < MIN_WINDOW_BATCH_SIZE:
        raise ValueError(
            "PERPLEXITY_MAX_WINDOW_BATCH_SIZE must be greater "
            "than or equal to PERPLEXITY_MIN_WINDOW_BATCH_SIZE. "
            f"Current minimum: {MIN_WINDOW_BATCH_SIZE}. "
            f"Current maximum: {MAX_WINDOW_BATCH_SIZE}."
        )

    if not 0.0 < GPU_MEMORY_UTILIZATION <= 1.0:
        raise ValueError(
            "PERPLEXITY_GPU_MEMORY_UTILIZATION must be greater "
            "than 0 and less than or equal to 1. "
            f"Current value: {GPU_MEMORY_UTILIZATION}"
        )

    if ESTIMATED_BYTES_PER_TOKEN <= 0:
        raise ValueError(
            "PERPLEXITY_ESTIMATED_BYTES_PER_TOKEN must be "
            "greater than 0. "
            f"Current value: {ESTIMATED_BYTES_PER_TOKEN}"
        )


def find_chunk_boundary(
    text: str,
    start: int,
    preferred_end: int,
) -> int:
    """
    Find a whitespace boundary for incremental tokenization.

    Args:
        text (str): Complete document text.
        start (int): Current fragment start.
        preferred_end (int): Preferred fragment end.

    Returns:
        int: Selected fragment end.
    """
    if preferred_end >= len(text):
        return len(text)

    for position in range(
        preferred_end,
        start,
        -1,
    ):
        if text[position].isspace():
            return position

    return preferred_end


def iter_text_chunks(
    text: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
) -> Iterator[str]:
    """
    Yield consecutive text fragments for incremental tokenization.

    The complete text already exists in host memory because it comes from one
    JSONL record. This iterator avoids materializing the complete token
    sequence.

    Args:
        text (str): Complete document text.
        chunk_size (int): Approximate characters per fragment.

    Yields:
        str: Consecutive text fragments.

    Raises:
        ValueError: If chunk_size is invalid.
    """
    if chunk_size <= 0:
        raise ValueError(
            "Text chunk size must be greater than 0. "
            f"Current value: {chunk_size}"
        )

    start = 0
    text_length = len(text)

    while start < text_length:
        preferred_end = min(
            start + chunk_size,
            text_length,
        )

        end = find_chunk_boundary(
            text=text,
            start=start,
            preferred_end=preferred_end,
        )

        if end <= start:
            end = preferred_end

        yield text[start:end]

        start = end


def get_special_token_boundaries(
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Determine special tokens added before and after a normal text sequence.

    This implementation uses the public tokenizer call interface instead of
    build_inputs_with_special_tokens(), which is not available in every
    tokenizer implementation, including some GemmaTokenizer versions.

    Args:
        tokenizer: Tokenizer associated with the language model.

    Returns:
        Tuple[List[int], List[int]]: Prefix and suffix special-token IDs.

    Raises:
        RuntimeError: If the content token sequence cannot be located inside
            the sequence containing special tokens.
    """
    sentinel_text = "test"

    content_ids = tokenizer(
        sentinel_text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    wrapped_ids = tokenizer(
        sentinel_text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    if not content_ids:
        raise RuntimeError(
            "Unable to determine special-token boundaries because "
            "the tokenizer produced no tokens for the sentinel text."
        )

    content_length = len(content_ids)
    content_position = None

    for position in range(
        len(wrapped_ids) - content_length + 1
    ):
        candidate = wrapped_ids[
            position:position + content_length
        ]

        if candidate == content_ids:
            content_position = position
            break

    if content_position is None:
        raise RuntimeError(
            "Unable to locate the regular token sequence inside the "
            "tokenized sequence containing special tokens. "
            f"Content IDs: {content_ids}. "
            f"Wrapped IDs: {wrapped_ids}."
        )

    prefix_ids = wrapped_ids[:content_position]

    suffix_ids = wrapped_ids[
        content_position + content_length:
    ]

    return prefix_ids, suffix_ids


def iter_token_ids(
    text: str,
    tokenizer,
) -> Iterator[int]:
    """
    Tokenize a document incrementally and yield token IDs.

    The complete token sequence is never stored as one list or tensor.

    Args:
        text (str): Complete document text.
        tokenizer: Model tokenizer.

    Yields:
        int: Consecutive token IDs.
    """
    prefix_ids, suffix_ids = get_special_token_boundaries(
        tokenizer
    )

    yield from prefix_ids

    for text_chunk in iter_text_chunks(
        text=text,
        chunk_size=TEXT_CHUNK_SIZE,
    ):
        encoding = tokenizer(
            text_chunk,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )

        token_ids = encoding.get(
            "input_ids",
            [],
        )

        yield from token_ids

    yield from suffix_ids


def validate_incremental_tokenization(
    text: str,
    tokenizer,
    chunk_size: int = TEXT_CHUNK_SIZE,
) -> None:
    """
    Compare incremental tokenization with complete tokenization.

    This function is intended only for tests with manageable sample texts.

    Args:
        text (str): Sample text.
        tokenizer: Tokenizer to validate.
        chunk_size (int): Fragment size used during the test.

    Raises:
        AssertionError: If token IDs differ.
    """
    complete_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    prefix_ids, suffix_ids = get_special_token_boundaries(
        tokenizer
    )

    incremental_ids: List[int] = list(prefix_ids)

    for text_chunk in iter_text_chunks(
        text=text,
        chunk_size=chunk_size,
    ):
        chunk_ids = tokenizer(
            text_chunk,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        incremental_ids.extend(chunk_ids)

    incremental_ids.extend(suffix_ids)

    if complete_ids == incremental_ids:
        return

    comparison_length = min(
        len(complete_ids),
        len(incremental_ids),
    )

    mismatch_position = next(
        (
            index
            for index in range(comparison_length)
            if complete_ids[index] != incremental_ids[index]
        ),
        comparison_length,
    )

    raise AssertionError(
        "Incremental tokenization differs from complete "
        "tokenization. "
        f"First mismatch position: {mismatch_position}. "
        f"Complete length: {len(complete_ids)}. "
        f"Incremental length: {len(incremental_ids)}. "
        f"Chunk size: {chunk_size}."
    )


def get_model_input_device(
    model,
    fallback_device: torch.device,
) -> torch.device:
    """
    Determine where input tensors should be placed.

    Args:
        model: Loaded causal language model.
        fallback_device (torch.device): Fallback device.

    Returns:
        torch.device: Input tensor device.
    """
    try:
        input_embeddings = model.get_input_embeddings()

        if input_embeddings is not None:
            return input_embeddings.weight.device

    except (AttributeError, RuntimeError):
        pass

    try:
        return next(model.parameters()).device

    except StopIteration:
        return fallback_device


def iter_document_windows(
    text: str,
    tokenizer,
    max_length: int,
) -> Iterator[List[int]]:
    """
    Yield overlapping token windows for one document.

    The generated windows reproduce the model-card slicing strategy:

        input_ids = encoded_tokens[start:start + max_length]
        start += STRIDE

    Token IDs are produced incrementally so the complete token sequence is
    never materialized in memory. Every yielded window is evaluated in full,
    including the overlap shared with adjacent windows.

    Args:
        text (str): Document text.
        tokenizer: Model tokenizer.
        max_length (int): Maximum number of tokens per window.

    Yields:
        List[int]: Token IDs for one complete or final partial window.
    """
    token_buffer: List[int] = []
    processed_windows = 0
    pending_new_tokens = 0

    for token_id in iter_token_ids(
        text=text,
        tokenizer=tokenizer,
    ):
        token_buffer.append(token_id)
        pending_new_tokens += 1

        if len(token_buffer) < max_length:
            continue

        yield list(token_buffer)

        processed_windows += 1
        pending_new_tokens = 0

        del token_buffer[:STRIDE]

    if token_buffer and (
        processed_windows == 0
        or pending_new_tokens > 0
    ):
        yield list(token_buffer)


def build_window_tasks(
    texts: Sequence[str],
    tokenizer,
    max_length: int,
) -> Iterator[WindowTask]:
    """
    Yield model-card-compatible inference windows for a record batch.

    Args:
        texts (Sequence[str]): Documents to evaluate.
        tokenizer: Model tokenizer.
        max_length (int): Maximum number of tokens per window.

    Yields:
        WindowTask: One complete sliding-window inference task.
    """
    for document_index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue

        for token_ids in iter_document_windows(
            text=text,
            tokenizer=tokenizer,
            max_length=max_length,
        ):
            yield WindowTask(
                document_index=document_index,
                token_ids=token_ids,
            )


def get_models_sharing_device(
    device: torch.device,
) -> int:
    """
    Count loaded models that use the same input device.

    Args:
        device (torch.device): Device to inspect.

    Returns:
        int: Number of resident models sharing the device.
    """
    with _MODEL_REGISTRY_LOCK:
        count = sum(
            1
            for loaded_model in _MODEL_REGISTRY.values()
            if loaded_model.device == device
        )

    return max(count, 1)


def get_adaptive_window_batch_size(
    device: torch.device,
    sequence_length: int,
) -> int:
    """
    Estimate a safe starting window batch size for current resources.

    The estimate uses current free GPU memory and is later corrected by
    automatic out-of-memory retries. A positive
    PERPLEXITY_WINDOW_BATCH_SIZE value acts as an explicit upper bound.
    Memory is divided between resident models sharing the same GPU. On CPU,
    a conservative batch size is used.

    Args:
        device (torch.device): Device used for inference.
        sequence_length (int): Longest sequence in the pending batch.

    Returns:
        int: Initial window batch size.
    """
    explicit_limit = (
        CONFIGURED_BATCH_SIZE
        if CONFIGURED_BATCH_SIZE > 0
        else MAX_WINDOW_BATCH_SIZE
    )

    maximum_batch_size = min(
        explicit_limit,
        MAX_WINDOW_BATCH_SIZE,
    )

    if device.type != "cuda":
        return max(
            MIN_WINDOW_BATCH_SIZE,
            min(maximum_batch_size, 2),
        )

    free_memory, _ = get_device_memory(
        device.index or 0
    )

    sharing_models = get_models_sharing_device(
        device
    )

    usable_memory = int(
        free_memory
        * GPU_MEMORY_UTILIZATION
        / sharing_models
    )

    estimated_memory_per_sequence = max(
        sequence_length * ESTIMATED_BYTES_PER_TOKEN,
        1,
    )

    estimated_batch_size = max(
        usable_memory // estimated_memory_per_sequence,
        MIN_WINDOW_BATCH_SIZE,
    )

    return int(
        max(
            MIN_WINDOW_BATCH_SIZE,
            min(
                estimated_batch_size,
                maximum_batch_size,
            ),
        )
    )


def prepare_window_batch(
    tasks: Sequence[WindowTask],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded model inputs and labels for a window batch.

    Every non-padding token is copied to the labels so each overlapping
    window is scored in full, matching the model-card recipe. Only padding
    positions are excluded from the loss with the value -100.

    Args:
        tasks (Sequence[WindowTask]): Window tasks to combine.
        pad_token_id (int): Token ID used for padding.
        device (torch.device): Device receiving tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Input IDs,
            attention mask, and labels.
    """
    maximum_length = max(
        len(task.token_ids)
        for task in tasks
    )

    batch_size = len(tasks)

    input_ids = torch.full(
        (batch_size, maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size, maximum_length),
        dtype=torch.long,
        device=device,
    )

    labels = torch.full(
        (batch_size, maximum_length),
        -100,
        dtype=torch.long,
        device=device,
    )

    for row_index, task in enumerate(tasks):
        sequence_length = len(task.token_ids)

        sequence_tensor = torch.tensor(
            task.token_ids,
            dtype=torch.long,
            device=device,
        )

        input_ids[
            row_index,
            :sequence_length,
        ] = sequence_tensor

        attention_mask[
            row_index,
            :sequence_length,
        ] = 1

        labels[
            row_index,
            :sequence_length,
        ] = sequence_tensor

    return input_ids, attention_mask, labels


def compute_window_batch_negative_log_likelihood(
    tasks: Sequence[WindowTask],
    loaded_model: LoadedModel,
) -> List[Tuple[int, float, int]]:
    """
    Compute model-card-compatible weighted loss for a window batch.

    The model card computes the average causal language-model loss for each
    window and multiplies it by the complete window length. This function
    reproduces that behavior independently for every padded sequence in the
    batch:

        weighted_nll = mean_window_loss * window_length

    The returned token count is also the complete window length, including
    overlap and the first token, exactly as in the published recipe.

    Args:
        tasks (Sequence[WindowTask]): Window tasks in the batch.
        loaded_model (LoadedModel): Model resources to use.

    Returns:
        List[Tuple[int, float, int]]: Document index, weighted negative
            log-likelihood, and complete window length for each task.

    Raises:
        RuntimeError: If the model returns a non-finite result.
    """
    tokenizer = loaded_model.tokenizer

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        raise RuntimeError(
            "The tokenizer has neither a pad token nor an EOS token."
        )

    input_ids, attention_mask, labels = prepare_window_batch(
        tasks=tasks,
        pad_token_id=int(pad_token_id),
        device=loaded_model.device,
    )

    stream_context = (
        torch.cuda.stream(loaded_model.stream)
        if loaded_model.stream is not None
        else torch.no_grad()
    )

    with stream_context:
        with torch.inference_mode():
            outputs = loaded_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            shift_logits = outputs.logits[
                :, :-1, :
            ].contiguous()

            shift_labels = labels[
                :, 1:
            ].contiguous()

            token_losses = functional.cross_entropy(
                shift_logits.float().view(
                    -1,
                    shift_logits.size(-1),
                ),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)

            valid_mask = shift_labels.ne(-100)

            loss_sums = (
                token_losses * valid_mask
            ).sum(dim=1)

            valid_token_counts = valid_mask.sum(dim=1)

            mean_losses = loss_sums / valid_token_counts.clamp_min(1)

            window_lengths = torch.tensor(
                [
                    len(task.token_ids)
                    for task in tasks
                ],
                dtype=torch.float32,
                device=loaded_model.device,
            )

            weighted_negative_log_likelihoods = (
                mean_losses * window_lengths
            )

    if loaded_model.stream is not None:
        loaded_model.stream.synchronize()

    result: List[Tuple[int, float, int]] = []

    for (
        task,
        weighted_negative_log_likelihood,
        valid_token_count,
    ) in zip(
        tasks,
        weighted_negative_log_likelihoods,
        valid_token_counts,
    ):
        valid_count = int(
            valid_token_count.detach().cpu().item()
        )

        if valid_count <= 0:
            continue

        nll_value = float(
            weighted_negative_log_likelihood
            .detach()
            .cpu()
            .item()
        )

        if not math.isfinite(nll_value):
            raise RuntimeError(
                "The model returned a non-finite weighted negative "
                f"log-likelihood value: {nll_value}"
            )

        result.append(
            (
                task.document_index,
                nll_value,
                len(task.token_ids),
            )
        )

    del input_ids
    del attention_mask
    del labels
    del outputs
    del shift_logits
    del shift_labels
    del token_losses
    del valid_mask
    del loss_sums
    del valid_token_counts
    del mean_losses
    del window_lengths
    del weighted_negative_log_likelihoods

    return result


def run_window_tasks_with_oom_recovery(
    tasks: Sequence[WindowTask],
    loaded_model: LoadedModel,
) -> List[Tuple[int, float, int]]:
    """
    Execute window tasks with automatic CUDA OOM batch reduction.

    Args:
        tasks (Sequence[WindowTask]): Pending inference windows.
        loaded_model (LoadedModel): Model resources to use.

    Returns:
        List[Tuple[int, float, int]]: Aggregation values per window.

    Raises:
        torch.cuda.OutOfMemoryError: If one window does not fit in memory.
    """
    if not tasks:
        return []

    longest_sequence = max(
        len(task.token_ids)
        for task in tasks
    )

    current_batch_size = get_adaptive_window_batch_size(
        device=loaded_model.device,
        sequence_length=longest_sequence,
    )

    results: List[Tuple[int, float, int]] = []
    start = 0

    while start < len(tasks):
        remaining = len(tasks) - start
        batch_size = min(
            current_batch_size,
            remaining,
        )

        current_tasks = tasks[
            start:start + batch_size
        ]

        try:
            batch_results = (
                compute_window_batch_negative_log_likelihood(
                    tasks=current_tasks,
                    loaded_model=loaded_model,
                )
            )

            results.extend(batch_results)
            start += batch_size

        except torch.cuda.OutOfMemoryError:
            if loaded_model.device.type == "cuda":
                with torch.cuda.device(loaded_model.device):
                    torch.cuda.empty_cache()

            if batch_size <= MIN_WINDOW_BATCH_SIZE:
                raise

            current_batch_size = max(
                batch_size // 2,
                MIN_WINDOW_BATCH_SIZE,
            )

            print(
                f"CUDA OOM for {loaded_model.metric_name}; "
                f"retrying with window batch size "
                f"{current_batch_size}."
            )

    return results


def compute_model_perplexity_batch(
    texts: List[str],
    loaded_model: LoadedModel,
) -> List[Optional[float]]:
    """
    Compute perplexity using the model-card sliding-window recipe.

    Windows from all documents are combined into true model batches. Every
    window is scored in full, including overlap with adjacent windows. The
    average loss for each window is multiplied by the complete window length,
    and the final perplexity is computed from the weighted average across all
    windows, matching the published model-card procedure.

    Tokenization remains incremental so memory usage does not scale with the
    complete tokenized document. CUDA out-of-memory errors automatically
    reduce the inference batch size.

    Args:
        texts (List[str]): Texts to evaluate.
        loaded_model (LoadedModel): Model resources to use.

    Returns:
        List[Optional[float]]: Values aligned with input texts.
    """
    tokenizer = loaded_model.tokenizer
    model = loaded_model.model

    max_length = get_effective_max_length(
        tokenizer,
        model,
    )

    validate_window_config(max_length)

    total_negative_log_likelihoods = [
        0.0
        for _ in texts
    ]

    total_window_tokens = [
        0
        for _ in texts
    ]

    task_buffer: List[WindowTask] = []

    initial_task_buffer_limit = max(
        MAX_WINDOW_BATCH_SIZE * 4,
        1,
    )

    def flush_tasks() -> None:
        if not task_buffer:
            return

        batch_results = run_window_tasks_with_oom_recovery(
            tasks=task_buffer,
            loaded_model=loaded_model,
        )

        for (
            document_index,
            weighted_negative_log_likelihood,
            window_length,
        ) in batch_results:
            total_negative_log_likelihoods[
                document_index
            ] += weighted_negative_log_likelihood

            total_window_tokens[
                document_index
            ] += window_length

        task_buffer.clear()

    with loaded_model.lock:
        for task in build_window_tasks(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
        ):
            task_buffer.append(task)

            if len(task_buffer) >= initial_task_buffer_limit:
                flush_tasks()

        flush_tasks()

    results: List[Optional[float]] = []

    for negative_log_likelihood, window_token_count in zip(
        total_negative_log_likelihoods,
        total_window_tokens,
    ):
        if window_token_count <= 0:
            results.append(None)
            continue

        average_negative_log_likelihood = (
            negative_log_likelihood
            / window_token_count
        )

        try:
            perplexity = math.exp(
                average_negative_log_likelihood
            )

        except OverflowError:
            perplexity = float("inf")

        results.append(float(perplexity))

    return results


def compute_perplexity_batch_for_metric(
    metric_name: str,
    texts: List[str],
) -> List[Optional[float]]:
    """
    Compute perplexity using the model associated with a metric.

    Args:
        metric_name (str): Metadata metric.
        texts (List[str]): Texts to evaluate.

    Returns:
        List[Optional[float]]: Values aligned with input texts.
    """
    loaded_model = get_model_for_metric(
        metric_name
    )

    return compute_model_perplexity_batch(
        texts=texts,
        loaded_model=loaded_model,
    )