# Heuristics Evaluation

Systemic evaluation of different heuristic filters

## Evaluation Methods

### Perplexity

Using each 

$$
% Standard Form
\text{Weighted Rel NLL} = \frac{\sum_{i=1}^{N} w_i \left( \ln(P_{\text{Coreguapa}, i}) - \ln(P_{\text{Tweets}, i}) \right)}{\sum_{i=1}^{N} w_i}
$$

$$
% Simplified Form
\text{Weighted Rel NLL} = \frac{\sum_{i=1}^{N} w_i \ln\left( \frac{P_{\text{Coreguapa}, i}}{P_{\text{Tweets}, i}} \right)}{\sum_{i=1}^{N} w_i}
$$

### MMLU

## Metrics

|Name|Description|Documents|Perplexity Score|Global-MMLU Accuracy|G


<!-- 
{
    "steps": [
        {
            "step_name": "Baseline",
            "description": "All corpus documents from all sources",
            "documents": 959936,
            "tokens": 38496060,
            "Weighted Relative NLL": -1.1462772054839092,
            "Global-MMLU Accuracy": 23.75
        },
        {
            "step_name": "Boilerplate Removal",
            "description": "Several filters mainly for the opus-all-en and fineweb-2 corpora filtering out boilerplate content",
            "documents": 954049,
            "tokens": 37901122,
            "Weighted Relative NLL": -1.1562888541232696,
            "Global-MMLU Accuracy": 23.75
        }
    ]
} -->

