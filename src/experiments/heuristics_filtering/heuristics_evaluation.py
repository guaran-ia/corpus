#imports
import pandas as pd
from pipeline.integration.utils.loader import load_jsonl
from pipeline.integration.utils.document import Document
from collections import Counter
from typing import Any, Dict, Iterator
import json
import numpy as np
import os
import sys
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm


SOURCE_DIRECTORY = "kuatia/1_1_0/3_pii_censoring/censored"
OUTPUT_DIRECTORY = "outputs/heuristics_exploration/eval"

num_cores = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(num_cores)
torch.set_num_threads(num_cores)


def iterate_directory(directory:str) -> Iterator[Document]:
    for file in os.listdir(directory):
        if file.endswith(".jsonl"):
            path = os.path.join(directory, file)
            documents = load_jsonl(str(path), field_map={"id":"id", "text":"text"}, load_metadata=True, metadata_fields="*")
            for doc in documents:
                yield doc

def get_dataframe_from_docs(data_path):
    documents = iterate_directory(data_path)

    rows = []

    for document in documents:

        row = {
            "id": document.id,
            "text": document.text
        }
        for field, value in document.metadata.items():
            if type(value) != "dict":
                row[field] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

#Metric that uses the perplexity score
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

def train_cpu_adapter(df:pd.DataFrame, output_dir:str, base_model_id="princeton-nlp/gemma-2-9b-it-SimPO"):

    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(output_dir) and os.path.exists(config_file):
        print(f"--- Found existing model at '{output_dir}'. Skipping training and loading checkpoint. ---")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        merged_model = AutoModelForCausalLM.from_pretrained(
            output_dir, 
            dtype=torch.float32, 
            device_map="cpu"
        )
        return merged_model, tokenizer
    
    # 1. Load the jsonl file
    dataset = Dataset.from_pandas(df[['text']])

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        dtype=torch.float32, 
        device_map="cpu"
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500,
        learning_rate=3e-4,
        use_cpu=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",  # Move dataset_text_field here!
        max_length=256,  # Move max_seq_length here as well!
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        #dataset_text_field="text",      # Points to your 'text' column
        #max_seq_length=256,             # Keeps training fast on CPU
        args=sft_config
    )

    trainer.train()
    
    # Save the adapter checkpoints
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model successfully saved to {output_dir}\n")

    return merged_model, tokenizer

def load_mmlu_jsonl(jsonl_filepath):
    """Loads MMLU evaluation data from a .jsonl file and formats it into the expected structure."""
    mmlu_data = []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    with open(jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Extract fields matching your schema
            question = item["question"]
            options = [
                item["option_a"],
                item["option_b"],
                item["option_c"],
                item["option_d"],
            ]
            raw_answer = str(item["answer"]).strip().upper()
            answer_idx = answer_map.get(raw_answer, 0)

            mmlu_data.append({
                "question": question,
                "options": options,
                "answer": answer_idx,
                "raw_answer": raw_answer,
            })

    print(f"Loaded {len(mmlu_data)} evaluation questions from {jsonl_filepath}")
    return mmlu_data

def evaluate_mmlu(model, tokenizer, mmlu_data):
    """Evaluates a model instance on multiple-choice data and logs detailed item-level results."""
    model.eval()
    option_labels = ["A", "B", "C", "D"]
    correct_count = 0
    item_results = []

    for idx, item in enumerate(tqdm(mmlu_data, desc="Evaluating MMLU")):
        # Build prompt format
        prompt = f"Porandu: {item['question']}\n"
        for i, opt in enumerate(item["options"]):
            prompt += f"{option_labels[i]}) {opt}\n"
        prompt += "Mbohovái:"

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
        option_log_probs = []

        # Calculate conditional log-probability for options: " A", " B", " C", " D"
        for label in option_labels:
            full_text = prompt + f" {label}"
            full_ids = tokenizer.encode(full_text, return_tensors="pt").to(
                "cpu"
            )
            option_token_ids = full_ids[:, prompt_ids.shape[1] :]

            with torch.no_grad():
                outputs = model(full_ids)
                logits = outputs.logits

            # Align shifted logits with next-token targets
            shift_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
            log_probs = F.log_softmax(shift_logits, dim=-1)

            target_log_probs = torch.gather(
                log_probs, dim=-1, index=option_token_ids.unsqueeze(-1)
            ).squeeze(-1)

            option_log_probs.append(target_log_probs.sum().item())

        # Determine predicted option
        pred_idx = int(torch.argmax(torch.tensor(option_log_probs)))
        is_correct = pred_idx == item["answer"]

        if is_correct:
            correct_count += 1

        # Track per-question breakdown for report analysis
        item_results.append({
            "question_id": idx,
            "question": item["question"],
            "ground_truth": item["raw_answer"],
            "predicted": option_labels[pred_idx],
            "is_correct": is_correct,
            "option_scores": {
                label: score
                for label, score in zip(option_labels, option_log_probs)
            },
        })

    accuracy = (correct_count / len(mmlu_data)) * 100
    return accuracy, item_results

def filter_and_eval(filters:list[dict]):
    report = {"steps":[]}

    mmlu_data_path = "src/experiments/heuristics_filtering/mmlu_eval/eval_tasks.jsonl"
    mmlu_data = load_mmlu_jsonl(mmlu_data_path)

    detailed_reports_path = f"{OUTPUT_DIRECTORY}/detailed_reports"
    os.makedirs(detailed_reports_path, exist_ok=True)

    print("Running Baseline Evaluation")
    #Read the source documents
    baseline_df = get_dataframe_from_docs(SOURCE_DIRECTORY)

    #Evaluate the source documents
    #Perplexity
    wrnll_baseline = weighted_relative_nll(baseline_df)

    #Global-MMLU
    #Train a baseline model
    baseline_model_dir = f"{OUTPUT_DIRECTORY}/models/baseline_qwen"
    os.makedirs(baseline_model_dir, exist_ok=True)

    baseline_model, baseline_tokenizer = train_cpu_adapter(baseline_df, baseline_model_dir)

    #Evaluate the model
    baseline_accuracy, baseline_details = evaluate_mmlu(baseline_model, baseline_tokenizer, mmlu_data)

    baseline_report = {
        "step_name":"Baseline",
        "description":"All corpus documents from all sources",
        "documents": len(baseline_df),
        "tokens": int(baseline_df['num_words_split'].sum()),
        "Weighted Relative NLL": wrnll_baseline,
        "Global-MMLU Accuracy": baseline_accuracy
    }

    report['steps'].append(baseline_report)

    baseline_detailed_path = f"{detailed_reports_path}/baseline_qwen.jsonl"

    with open(baseline_detailed_path, "w", encoding="utf-8") as f:
        for line in baseline_details:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    #Apply filters
    filtered_df = baseline_df.copy()

    for i, filter in enumerate(filters):

        filter_func = filter.get('callable', None)
        filter_name = filter.get('name', f'filter_{i}')

        print(f"Evaluating Filter: {filter_name}")

        if not filter_func:
            print("Invalid filter function")
            break
        
        filtered_df = filter_func(filtered_df)

        #Perplexity
        wrnll = weighted_relative_nll(filtered_df)

        #Global-MMLU
        model_dir = f"{OUTPUT_DIRECTORY}/models/{"_".join(filter_name.lower().split())}_qwen"
        os.makedirs(model_dir, exist_ok=True)

        model, tokenizer = train_cpu_adapter(filtered_df, model_dir)

        accuracy, details = evaluate_mmlu(model, tokenizer, mmlu_data)

        filter_report = {
            "step_name":filter_name,
            "description":filter.get("description", ""),
            "documents": len(filtered_df),
            "tokens": int(filtered_df['num_words_split'].sum()),
            "Weighted Relative NLL": wrnll,
            "Global-MMLU Accuracy": accuracy
        }

        report['steps'].append(filter_report)

        detailed_path = f"{detailed_reports_path}/{"_".join(filter_name.lower().split())}.jsonl"
        
        with open(detailed_path, "w", encoding="utf-8") as f:
            for line in details:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    report_path = f"{OUTPUT_DIRECTORY}/report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def apply_boilerplate_filters(df:pd.DataFrame) -> pd.DataFrame:
    """Applies filters that are meant to erase boilerplate content"""

    #Remove documents with legal phrases
    df = df[~(df['count_sentences_with_legal_phrases'] > 0)]

    #Remove documents with curly brackets for the fineweb-2 and opus-all-en corpora
    df = df[~((df['count_sentences_with_curly_bracket'] > 0) & (df['corpus'] == 'opus-all-en') & (df['language_score'] < 0.9965))]
    df = df[~((df['count_sentences_with_curly_bracket'] > 0) & (df['corpus'] == 'fineweb-2'))]

    #Remove documents with javascript for the same corpora
    df = df[~((df['count_sentences_with_javascript'] > 0) & (df['corpus'] == 'opus-all-en') & (df['language_score'] < 0.95))]
    df = df[~((df['count_sentences_with_javascript'] > 0) & (df['corpus'] == 'fineweb-2'))]

    #Remove content
    df = df[~((df['ratio_symbols_to_words'] > 3) & (df['corpus'] == 'opus-all-en') & (df['language_score'] < 1))]

    #Remove documents with low mean word length
    #TODO: also test with high word length
    corpus_to_remove = ['opus-all-en', 'opus', 'gua_spa', 'belele', 'FinePDF', 'josa']
    df = df[~((df['mean_word_length'] < 3.0) & (df['corpus'].isin(corpus_to_remove)))]

    return df


if __name__ == "__main__":
    filters = [
        {
            "name": "Boilerplate Removal",
            "description": "Several filters mainly for the opus-all-en and fineweb-2 corpora filtering out boilerplate content",
            "callable":apply_boilerplate_filters
        }
    ]

    filter_and_eval(filters)