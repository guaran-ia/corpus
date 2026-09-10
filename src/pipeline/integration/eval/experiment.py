import os
import shutil
import json
import yaml
import tempfile
from typing import Dict, Any, List, Callable, Optional, Union

import lm_eval
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.utils import handle_non_serializable

from .. import Pipeline, PipelineStep
from .fine_tune import ModelFineTuner
from .experiment_loader import load_config
from .experiment_report import ExperimentReport


class Experiment:
    """Executes an end-to-end NLP data processing, model fine-tuning, and evaluation pipeline.

    Processes documents through a defined sequence of pipeline steps inside a temporary sandbox directory,
    evaluates dataset-level metrics on the transformed data, optionally fine-tunes a causal language model
    on the pipeline's output, and evaluates the resulting model using `lm-eval-harness`.

    Attributes:
        experiment_name (str): Unique name identifier for the experiment run.
        input_directory (str): Root path containing raw input dataset files.
        output_directory (str): Destination directory for reports, saved models, and metrics.
        steps (List[Union[PipelineStep, Dict[str, Any]]]): List of pipeline transformation steps.
        dataset_metrics (Optional[Dict[str, Callable[[str], Any]]]): Named callable metric functions evaluated on pipeline output.
        model_trainer (Optional[ModelFineTuner]): ModelFineTuner instance configured for fine-tuning.
        lm_eval_tasks (Optional[List[str]]): List of lm-eval task names to run.
        lm_eval_include_path (Optional[str]): Local file path containing custom YAML task definitions for lm-eval.
    """

    def __init__(
        self,
        name: str,
        input_directory: str,
        output_directory: str,
        steps: List[Union[PipelineStep, Dict[str, Any]]],
        dataset_metrics: Optional[Dict[str, Callable[[str], Any]]] = None,
        model_trainer: Optional[Union[ModelFineTuner, Dict[str, Any]]] = None,
        lm_eval_tasks: Optional[Union[List[str], str]] = None,
        lm_eval_include_path: Optional[str] = None,
    ):
        """Initializes the Experiment runner and validates parameter combinations.

        Args:
            name (str): Name of the experiment.
            input_directory (str): Directory containing source document files.
            output_directory (str): Directory where reports and artifacts will be saved.
            steps (List[Union[PipelineStep, Dict[str, Any]]]): Pipeline execution steps.
            dataset_metrics (Optional[Dict[str, Callable[[str], Any]]]): Dictionary mapping metric names 
                to functions that accept `(pipeline_output_path: str)` and return evaluation results.
            model_trainer (Optional[Union[ModelFineTuner, Dict[str, Any]]]): ModelFineTuner instance or dictionary payload.
            lm_eval_tasks (Optional[Union[List[str], str]]): Single task name, list of task names, or comma-separated task string.
            lm_eval_include_path (Optional[str]): Path to local folder containing custom YAML task configs for lm-eval.

        Raises:
            ValueError: If neither `dataset_metrics` nor `lm_eval_tasks` are provided, or if `lm_eval_tasks` is defined without a `model_trainer`.
        """
        if not dataset_metrics and not lm_eval_tasks:
            raise ValueError("Found no evaluation parameters. Must provide at least 'dataset_metrics' or 'lm_eval_tasks'.")

        if lm_eval_tasks and not model_trainer:
            raise ValueError("`lm_eval_tasks` were specified, but no `model_trainer` was configured.")

        self.experiment_name = name
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.steps = steps
        self.dataset_metrics = dataset_metrics or {}
        
        # Parse lm_eval tasks
        if isinstance(lm_eval_tasks, str):
            self.lm_eval_tasks = [t.strip() for t in lm_eval_tasks.split(",")]
        else:
            self.lm_eval_tasks = lm_eval_tasks or []

        self.lm_eval_include_path = lm_eval_include_path

        # Setup model trainer if provided
        self.model_trainer = None
        if model_trainer:
            if isinstance(model_trainer, ModelFineTuner):
                self.model_trainer = model_trainer
            elif isinstance(model_trainer, dict):
                # Target path where fine-tuned model will be stored
                target_model_dir = os.path.join(self.output_directory, f"{self.experiment_name}_model")
                trainer_kwargs = model_trainer.copy()
                trainer_kwargs.setdefault("input_directory", "")  # Will be set dynamically after pipeline execution
                trainer_kwargs["output_directory"] = target_model_dir
                self.model_trainer = ModelFineTuner.from_dict(trainer_kwargs)

        os.makedirs(self.output_directory, exist_ok=True)
        self.experiment_report = ExperimentReport(name=self.name)

    @property
    def name(self) -> str:
        return self.experiment_name

    @property
    def report(self) -> ExperimentReport:
        return self.experiment_report

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Experiment":
        """Instantiates an Experiment from a configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing experiment instantiation arguments.

        Returns:
            Experiment: Instantiated experiment instance.
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Experiment":
        """Factory method to instantiate an Experiment directly from a YAML config file.

        Args:
            config_path (str): Path to the target YAML experiment file.

        Returns:
            Experiment: Instantiated experiment instance.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Experiment config file not found at: '{config_path}'")

        # Fallback to load_config if provided, or safe_load
        try:
            config = load_config(config_path)
        except (NameError, TypeError):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        return cls.from_dict(config)

    def execute(self) -> ExperimentReport:
        """Runs the entire experimental workflow (Pipeline -> Metrics -> Fine-Tuning -> LM-Eval).

        Returns:
            ExperimentReport: Complete aggregated report of pipeline steps, metrics, and evaluation results.
        """
        # Execute the pipeline inside a temporary directory
        with tempfile.TemporaryDirectory(suffix="_experiment", dir=".") as temp_directory:
            print(f"--- Running Pipeline in Sandbox: '{temp_directory}' ---")
            pipeline = Pipeline(
                steps=self.steps,
                input_directory=self.input_directory,
                output_directory=temp_directory,
            )

            pipeline_output_path = pipeline.run()

            # 1. Copy Pipeline Report
            pipeline_report_path = os.path.join(temp_directory, "report.json")
            if os.path.exists(pipeline_report_path):
                dest_pipeline_report = os.path.join(self.output_directory, "pipeline_report.json")
                shutil.copy2(pipeline_report_path, dest_pipeline_report)
                self.experiment_report.pipeline_report = os.path.relpath(dest_pipeline_report, self.output_directory)

            # 2. Copy Individual Step Reports
            step_reports_dir = os.path.join(self.output_directory, "step_reports")
            for item in os.listdir(temp_directory):
                step_dir = os.path.join(temp_directory, item)
                if os.path.isdir(step_dir):
                    report_path = os.path.join(step_dir, "report.json")
                    if os.path.exists(report_path):
                        os.makedirs(step_reports_dir, exist_ok=True)
                        dest_step_report = os.path.join(step_reports_dir, f"{item}_report.json")
                        shutil.copy2(report_path, dest_step_report)
                        self.experiment_report.step_reports[item] = os.path.relpath(dest_step_report, self.output_directory)

            # 3. Execute Dataset Metrics
            if self.dataset_metrics:
                print("--- Calculating Dataset Metrics ---")
                for metric_name, func in self.dataset_metrics.items():
                    self.experiment_report.dataset_metrics[metric_name] = func(pipeline_output_path)

            # 4. Fine-Tune Model & Run LM-Eval Harness (Only if tasks and trainer are present)
            if self.model_trainer and self.lm_eval_tasks:
                print("--- Fine-Tuning Model on Pipeline Output ---")
                self.model_trainer.input_directory = pipeline_output_path
                fine_tuned_model_dir = self.model_trainer.train()

                print("--- Running LM Evaluation Harness ---")
                use_cpu = getattr(self.model_trainer, "use_cpu", False)
                eval_device = "cpu" if use_cpu else "cuda"

                model_args = {
                    "pretrained": fine_tuned_model_dir,
                    "dtype": "float32" if use_cpu else "bfloat16",
                }
                if not use_cpu:
                    model_args["device_map"] = "auto"

                eval_config = EvaluatorConfig(
                    model="hf",
                    model_args=model_args,
                    include_path=self.lm_eval_include_path or "",
                    tasks=self.lm_eval_tasks,
                    device=eval_device,
                    output_path=self.output_directory,
                )

                task_manager = eval_config.process_tasks()

                results = lm_eval.simple_evaluate(
                    model=eval_config.model,
                    model_args=eval_config.model_args,
                    task_manager=task_manager,
                    tasks=eval_config.tasks,
                    device=eval_config.device,
                )

                if results and "results" in results:
                    self.experiment_report.lm_eval_results = results["results"]

                # Save raw lm_eval results
                eval_out_file = os.path.join(self.output_directory, "lm_eval_results.json")
                with open(eval_out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, default=handle_non_serializable, indent=2)

        # 5. Save Final Aggregated Experiment Report
        report_path = os.path.join(self.output_directory, "experiment_report.json")
        print(f"Writing Final Experiment Report to: '{report_path}'")
        self.experiment_report.write_json(report_path)

        return self.experiment_report