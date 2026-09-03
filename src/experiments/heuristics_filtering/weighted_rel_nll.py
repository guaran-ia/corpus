import numpy as np
import pandas as pd


#Uses the perplexity score 
def weighted_relative_nll(df:pd.DataFrame):
    if len(df) == 0:
        return np.nan
    nll_high = np.log(df["coreguapa_perplexity"])
    nll_low = np.log(df["tweets_perplexity"])
    rel_nll = nll_high - nll_low

    total_tokens = df["num_words_split"].sum()
    if total_tokens == 0:
        return np.nan
    return (df["num_words_split"] * rel_nll).sum() / total_tokens