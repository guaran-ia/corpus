import os
import shutil
import tempfile
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from src.pipeline.integration import Pipeline

class Experiment:
    def __init__(self, name: str, pipeline: Pipeline, eval_config: dict, config_path: str = None):
        self.name = name
        self.pipeline = pipeline
        self.eval_config = eval_config or {}
        self.config_path = config_path

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Experiment":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        exp_name = config.get("experiment", {}).get("name", "unnamed_experiment")
        pipeline = Pipeline.from_yaml(yaml_path)
        eval_config = config.get("evaluation", {})

        return cls(name=exp_name, pipeline=pipeline, eval_config=eval_config, config_path=yaml_path)

    def _fine_tune_model(self, jsonl_dir: str, adapter_output_dir: str) -> str:
        """Trains adapter directly on JSONL files using Hugging Face datasets (no Pandas)."""
        base_model_id = self.eval_config.get("base_model_id", "google/gemma-2-9b-it")
        device = self.eval_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset directly from JSONL directory without bringing everything into RAM
        dataset = load_dataset("json", data_files=os.path.join(jsonl_dir, "*.jsonl"), split="train")

        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

        sft_config = SFTConfig(
            output_dir=adapter_output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=self.eval_config.get("max_train_steps", 500),
            learning_rate=3e-4,
            use_cpu=(device == "cpu"),
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_length=256,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config
        )

        trainer.train()

        # Save merged model locally for lm-eval
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)

        # Free GPU VRAM prior to evaluation
        del model
        del merged_model
        torch.cuda.empty_cache()

        return adapter_output_dir

    def evaluate(self, use_temp_dir: bool = True) -> dict:
        """Runs pipeline, calculates NLL metric, fine-tunes model, and evaluates with lm-eval."""
        results = {"experiment_name": self.name}
        
        # Determine context for disk space management
        temp_dir_obj = tempfile.TemporaryDirectory() if use_temp_dir else None
        target_dir = temp_dir_obj.name if temp_dir_obj else os.path.join("outputs", self.name)

        try:
            # Set pipeline output path dynamically
            self.pipeline.output_directory = os.path.join(target_dir, "pipeline_out")
            
            print(f"--- Running Pipeline for Experiment: {self.name} ---")
            self.pipeline.run()

            # The final directory where filtered jsonl documents live
            final_jsonl_dir = os.path.join(
                self.pipeline.output_directory,
                self.pipeline.report.output_directory
            )

            # 1. Calculate Weighted Relative NLL Metric
            print("--- Calculating Weighted Relative NLL ---")
            nll_metric = calculate_weighted_relative_nll_streaming(final_jsonl_dir)
            results["weighted_relative_nll"] = nll_metric

            # 2. Fine-tune model
            print("--- Fine-Tuning Model ---")
            model_checkpoint_dir = os.path.join(target_dir, "fine_tuned_model")
            self._fine_tune_model(final_jsonl_dir, model_checkpoint_dir)

            # 3. Evaluate using lm-eval
            print("--- Running LM Evaluation ---")
            lm_eval_tasks = self.eval_config.get("lm_eval_tasks", [])
            if lm_eval_tasks:
                lm_results = run_lm_evaluation(
                    model_path=model_checkpoint_dir,
                    tasks=lm_eval_tasks,
                    custom_tasks_dir=self.eval_config.get("custom_tasks_dir"),
                    device=self.eval_config.get("device", "cuda")
                )
                results["lm_eval"] = lm_results

        finally:
            # Clean up disk space automatically if using temporary directories
            if temp_dir_obj:
                temp_dir_obj.cleanup()

        return results