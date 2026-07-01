from __future__ import annotations

import gc
import math
import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COREGUAPA_MODEL_ID = "guaran-ia/coreguapa-lm"
GNTWEETS_MODEL_ID = "guaran-ia/gntweets-lm"

MAX_LENGTH = int(os.getenv("PERPLEXITY_MAX_LENGTH", "8192"))
STRIDE = int(os.getenv("PERPLEXITY_STRIDE", "4096"))

HF_HOME = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
CACHE_DIR = os.getenv("HF_HUB_CACHE", f"{HF_HOME}/hub")
LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"

_MODEL = None
_TOKENIZER = None
_CURRENT_MODEL_ID = None

MODEL_BY_METRIC = {
    "coreguapa_perplexity": COREGUAPA_MODEL_ID,
    "tweets_perplexity": GNTWEETS_MODEL_ID,
}


def get_device() -> str:
    """
    Return the device that should be used for model inference.

    Args:
        None.

    Returns:
        str: "cuda" when a CUDA GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def unload_model() -> None:
    """
    Unload the currently cached model and tokenizer from memory.

    Args:
        None.

    Returns:
        None.
    """
    global _MODEL, _TOKENIZER, _CURRENT_MODEL_ID

    _MODEL = None
    _TOKENIZER = None
    _CURRENT_MODEL_ID = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model(model_id: str):
    """
    Load a tokenizer and causal language model from Hugging Face.

    Args:
        model_id (str): Hugging Face model identifier to load.

    Returns:
        tuple: A tuple containing the tokenizer and the loaded model.

    Raises:
        RuntimeError: If the model or tokenizer cannot be loaded.
    """
    print(f"Loading model {model_id} using cache_dir={CACHE_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={},
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
        )

        model.eval()

        return tokenizer, model

    except Exception as error:
        unload_model()

        raise RuntimeError(
            f"Failed to load model {model_id} from cache_dir={CACHE_DIR}. "
            f"local_files_only={LOCAL_FILES_ONLY}. Original error: {error}"
        ) from error


def get_model_for_metric(metric_name: str):
    """
    Return the model and tokenizer associated with a perplexity metric.

    Args:
        metric_name (str): Name of the metric to compute. Supported values are
            "coreguapa_perplexity" and "tweets_perplexity".

    Returns:
        tuple: A tuple containing the tokenizer, model, and selected device.

    Raises:
        ValueError: If the metric name is not supported.
        RuntimeError: If the required model cannot be loaded.
    """
    global _MODEL, _TOKENIZER, _CURRENT_MODEL_ID

    if metric_name not in MODEL_BY_METRIC:
        raise ValueError(f"Unsupported metric name: {metric_name}")

    model_id = MODEL_BY_METRIC[metric_name]

    if _CURRENT_MODEL_ID != model_id:
        unload_model()
        _TOKENIZER, _MODEL = load_model(model_id)
        _CURRENT_MODEL_ID = model_id

    return _TOKENIZER, _MODEL, get_device()


def get_effective_max_length(tokenizer) -> int:
    """
    Resolve the maximum context length used for sliding-window perplexity.

    Args:
        tokenizer: Tokenizer associated with the language model.

    Returns:
        int: Effective maximum window length.
    """
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


def validate_window_config(max_length: int) -> None:
    """
    Validate the sliding-window configuration.

    Args:
        max_length (int): Effective maximum context length.

    Returns:
        None.

    Raises:
        ValueError: If STRIDE or max_length are invalid.
    """
    if max_length <= 1:
        raise ValueError(
            f"PERPLEXITY_MAX_LENGTH must be greater than 1. "
            f"Current value: {max_length}"
        )

    if STRIDE <= 0:
        raise ValueError(
            f"PERPLEXITY_STRIDE must be greater than 0. "
            f"Current value: {STRIDE}"
        )


def compute_full_text_perplexity(
    text: str,
    tokenizer,
    model,
    device: str,
) -> Optional[float]:
    """
    Compute perplexity over the full text using sliding token windows.

    Args:
        text (str): Full document text to evaluate.
        tokenizer: Tokenizer associated with the language model.
        model: Loaded causal language model.
        device (str): Device where tensors should be moved.

    Returns:
        Optional[float]: Perplexity computed over all available tokens, or None
            for empty texts.
    """
    if not text or not text.strip():
        return None

    max_length = get_effective_max_length(tokenizer)
    validate_window_config(max_length)

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = encodings["input_ids"].to(device)
    sequence_length = input_ids.size(1)

    if sequence_length <= 1:
        return None

    negative_log_likelihoods = []
    previous_end_location = 0

    for begin_location in range(
        0,
        sequence_length,
        STRIDE,
    ):
        end_location = min(
            begin_location + max_length,
            sequence_length,
        )

        target_length = end_location - previous_end_location

        input_ids_window = input_ids[
            :,
            begin_location:end_location,
        ]

        target_ids = input_ids_window.clone()

        if target_length < input_ids_window.size(1):
            target_ids[
                :,
                :-target_length,
            ] = -100

        with torch.inference_mode():
            outputs = model(
                input_ids_window,
                labels=target_ids,
            )

            negative_log_likelihood = (
                outputs.loss
                * target_length
            )

        negative_log_likelihoods.append(
            negative_log_likelihood.detach().cpu()
        )

        previous_end_location = end_location

        if end_location == sequence_length:
            break

    total_negative_log_likelihood = torch.stack(
        negative_log_likelihoods
    ).sum()

    total_tokens = sequence_length - 1

    if total_tokens <= 0:
        return None

    average_negative_log_likelihood = (
        total_negative_log_likelihood
        / total_tokens
    )

    return float(
        math.exp(
            average_negative_log_likelihood.item()
        )
    )


def compute_model_perplexity_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
) -> List[Optional[float]]:
    """
    Compute perplexity values for a batch of full texts.

    Args:
        texts (List[str]): Texts to evaluate.
        tokenizer: Tokenizer associated with the language model.
        model: Loaded causal language model.
        device (str): Device where tensors should be moved.

    Returns:
        List[Optional[float]]: Perplexity values aligned with the input texts.
            Empty texts return None.
    """
    results: List[Optional[float]] = []

    try:
        for text in texts:
            perplexity = compute_full_text_perplexity(
                text,
                tokenizer,
                model,
                device,
            )

            results.append(perplexity)

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise

    return results


def compute_perplexity_batch_for_metric(
    metric_name: str,
    texts: List[str],
) -> List[Optional[float]]:
    """
    Compute perplexity values for a batch of texts using the metric model.

    Args:
        metric_name (str): Metadata metric to compute.
        texts (List[str]): Texts to evaluate.

    Returns:
        List[Optional[float]]: Perplexity values aligned with the input texts.
    """
    tokenizer, model, device = get_model_for_metric(metric_name)

    return compute_model_perplexity_batch(
        texts,
        tokenizer,
        model,
        device,
    )