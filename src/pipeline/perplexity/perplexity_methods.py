from __future__ import annotations

import math

from typing import Optional

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

COREGUAPA_MODEL_ID = "guaran-ia/coreguapa-lm"
GNTWEETS_MODEL_ID = "guaran-ia/gntweets-lm"

MAX_LENGTH = 2048

_COREGUAPA_MODEL = None
_COREGUAPA_TOKENIZER = None

_GNTWEETS_MODEL = None
_GNTWEETS_TOKENIZER = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_model(
    model_id: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        extra_special_tokens={},
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    device = model.device

    return tokenizer, model, device


def get_coreguapa_model():
    global _COREGUAPA_MODEL
    global _COREGUAPA_TOKENIZER

    if (
        _COREGUAPA_MODEL is None
        or _COREGUAPA_TOKENIZER is None
    ):
        (
            _COREGUAPA_TOKENIZER,
            _COREGUAPA_MODEL,
            _,
        ) = load_model(
            COREGUAPA_MODEL_ID
        )

    return (
        _COREGUAPA_TOKENIZER,
        _COREGUAPA_MODEL,
        get_device(),
    )


def get_gntweets_model():
    global _GNTWEETS_MODEL
    global _GNTWEETS_TOKENIZER

    if (
        _GNTWEETS_MODEL is None
        or _GNTWEETS_TOKENIZER is None
    ):
        (
            _GNTWEETS_TOKENIZER,
            _GNTWEETS_MODEL,
            _,
        ) = load_model(
            GNTWEETS_MODEL_ID
        )

    return (
        _GNTWEETS_TOKENIZER,
        _GNTWEETS_MODEL,
        get_device(),
    )


def compute_model_perplexity(
    text: str,
    tokenizer,
    model,
    device: str,
) -> Optional[float]:

    if not text.strip():
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )

        loss = outputs.loss

    return math.exp(
        loss.item()
    )


def compute_coreguapa_perplexity(
    text: str,
) -> Optional[float]:

    tokenizer, model, device = get_coreguapa_model()

    return compute_model_perplexity(
        text,
        tokenizer,
        model,
        device,
    )


def compute_tweets_perplexity(
    text: str,
) -> Optional[float]:

    tokenizer, model, device = get_gntweets_model()

    return compute_model_perplexity(
        text,
        tokenizer,
        model,
        device,
    )