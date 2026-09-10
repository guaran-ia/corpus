import os
import glob
import math
from src.pipeline.integration import load_jsonl


def weighted_relative_nll(path: str) -> float:
    """Calculates the token-weighted relative NLL for all Documents in the given path.

    Args:
        path (str): File path to a .jsonl file or a directory containing .jsonl files.

    Returns:
        float: The weighted relative NLL value, or NaN if no valid tokens/documents are found.
    """
    # 1. Resolve path into a list of target .jsonl file paths
    if os.path.isfile(path):
        jsonl_files = [path]
    elif os.path.isdir(path):
        jsonl_files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True) or \
                      glob.glob(os.path.join(path, "*.jsonl"))
    else:
        raise FileNotFoundError(f"Dataset path not found: '{path}'")

    if not jsonl_files:
        return float("nan")

    total_weighted_rel_nll = 0.0
    total_tokens = 0

    # 2. Iterate through each file and Document stream
    for file_path in jsonl_files:
        for doc in load_jsonl(file_path, field_map = {"text":"text", "id":"id"}, load_metadata = True, generate_id=False):
            # Access doc.metadata dictionary safely
            meta = getattr(doc, "metadata", {}) or {}

            coreguapa_ppl = meta.get("coreguapa_perplexity")
            tweets_ppl = meta.get("tweets_perplexity")
            num_words = meta.get("num_words_split", 0)

            # Skip entries missing perplexity or token values
            if coreguapa_ppl is None or tweets_ppl is None or coreguapa_ppl <= 0 or tweets_ppl <= 0 or num_words <= 0:
                continue

            # Calculate relative NLL: log(hq_perplexity) - log(lq_perplexity)
            rel_nll = math.log(coreguapa_ppl) - math.log(tweets_ppl)

            total_weighted_rel_nll += num_words * rel_nll
            total_tokens += num_words

    if total_tokens == 0:
        return float("nan")

    return float(total_weighted_rel_nll / total_tokens)