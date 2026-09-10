import os
import torch
import glob
import gc
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from typing import Dict, Union, Any


class ModelFineTuner:
    """Fine-tunes a Causal Language Model using LoRA adapters on JSONL datasets.

    Loads training datasets, sets up hardware precision (bfloat16/float32), fine-tunes
    a PEFT LoRA adapter, merges the adapter into the base model, and clears GPU memory.

    Attributes:
        input_directory (str): Path to directory containing target `.jsonl` files.
        output_directory (str): Destination path for saving merged model and tokenizer.
        sft_config (SFTConfig): Supervised Fine-Tuning configuration arguments.
        base_model_id (str): HuggingFace hub model ID or local directory path.
        peft_config (LoraConfig): Configuration for PEFT LoRA parameters.
        use_cpu (bool): Whether to force CPU execution over CUDA GPU.
    """
    def __init__(
            self, input_directory: str, output_directory: str, sft_config: Union[SFTConfig, Dict[str, Any]],
            base_model_id:str = "princeton-nlp/gemma-2-9b-it-SimPO", peft_config: Union[LoraConfig, Dict[str, Any]] = None,
            use_cpu: bool = False
        ):
        
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.use_cpu = use_cpu

        if use_cpu:
            self.device_map = "cpu"
            self.torch_dtype = torch.float32
        else:
            self.device_map = None  # Trainer manages GPU device placement
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


        #Parse the SFT config and fix any potential mismatch output issues
        self.sft_config = self._parse_sft_config(sft_config)

        self.sft_config.output_dir = self.output_directory
        self.sft_config.use_cpu = use_cpu

        if not use_cpu:
            sft_config.bf16 = (self.torch_dtype == torch.bfloat16)
            sft_config.fp16 = (self.torch_dtype == torch.float16)

        self.base_model_id = base_model_id

        self.peft_config = self._parse_peft_config(peft_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelFineTuner":
        """Instantiates ModelFineTuner from a Python dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing class arguments.

        Returns:
            ModelFineTuner: Configured instance of ModelFineTuner.
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelFineTuner":
        """Instantiates ModelFineTuner from a YAML configuration file.

        Args:
            yaml_path (str): Path to the `.yaml` configuration file.

        Returns:
            ModelFineTuner: Configured instance of ModelFineTuner.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration YAML file not found at: '{yaml_path}'")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)
    
    @classmethod
    def _parse_sft_config(cls, sft_config):
        parsed_config = None

        if isinstance(sft_config, SFTConfig):
            parsed_config = sft_config
        elif isinstance(sft_config, dict):
            parsed_config = SFTConfig(**sft_config)

        return parsed_config

    @classmethod
    def _parse_peft_config(cls, peft_config):
        parsed_config = None

        if not peft_config:
            parsed_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if isinstance(peft_config, LoraConfig):
            parsed_config = peft_config
        elif isinstance(peft_config, dict):
            parsed_config = LoraConfig(**peft_config)

        return parsed_config

    def train(self):
        config_file = os.path.join(self.output_directory, "config.json")

        if os.path.exists(self.output_directory) and os.path.exists(config_file):
            print(f"--- Found existing model at '{self.output_directory}'. Skipping training. ---")

        #Load the dataset
        jsonl_files = glob.glob(os.path.join(self.input_directory, "**", "*.jsonl"), recursive=True) or \
                  glob.glob(os.path.join(self.input_directory, "*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl dataset files found in directory: '{self.input_directory}'")
    
        dataset = load_dataset("json", data_files={"train": jsonl_files})

        #Setup the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            peft_config=self.peft_config,
            processing_class=tokenizer,
            args=self.sft_config,
        )

        print("--- Starting Model Fine-Tuning ---")
        trainer.train()

        print("--- Merging LoRA adapters with base model ---")
        if isinstance(trainer.model, PeftModel):
            merged_model = trainer.model.merge_and_unload()
        else:
            merged_model = trainer.model

        os.makedirs(self.output_directory, exist_ok=True)
        merged_model.save_pretrained(self.output_directory)
        tokenizer.save_pretrained(self.output_directory)

        print(f"Merged model and tokenizer saved to: {self.output_directory}\n")

        #MEMORY CLEANUP
        del trainer
        del model
        del merged_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()