from .pipeline_steps.pipeline_step import PipelineStep
from .pipeline_steps import STEP_REGISTRY
from .pipeline_report import PipelineReport
import os
from datetime import datetime
from typing import List, Dict, Any, Union
from .system_info import collect_system_info
import time
import psutil
import yaml

class Pipeline():
    
    """
    Executes a sequence of Pipeline Steps.
    
    """
    def __init__(self, steps:List[Union[type[PipelineStep], Dict[str, Any]]], input_directory:str, output_directory:str, start: int = 1):
        self.steps = self._parse_steps(steps)
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.report = PipelineReport(total_steps=len(steps))
        self.start = start

    @classmethod
    def _parse_steps(cls, steps: List[Union[PipelineStep, Dict[str, Any]]]) -> List[type[PipelineStep]]:
        """Ensures all step items are instantiated child classes of PipelineStep."""

        parsed_steps = []

        for step in steps:
            if isinstance(step, PipelineStep):
                parsed_steps.append(step)
            elif isinstance(step, dict):
                step_type_name = step["step_type"]
                if step_type_name not in STEP_REGISTRY:
                    raise ValueError(
                        f"Unknown step type '{step_type_name}'. Available steps: {list(STEP_REGISTRY.keys())}"
                    )
                step_class = STEP_REGISTRY[step_type_name]
                step_args = step.get("args")
                parsed_steps.append(step_class.from_dict(step_args))
            else:
                raise TypeError(
                    f"Step must be either a PipelineStep instance or a dictionary, got {type(step)}"
                )
        return parsed_steps

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Pipeline":
        """Instantiates a Pipeline from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Pipeline":
        """
        Instantiates a Pipeline directly from a configuration dictionary.
        """
        steps_config = config.get("steps", [])

        return cls(
            steps=steps_config,  # Parsed via _parse_steps in __init__
            input_directory=config.get("input_directory"),
            output_directory=config.get("output_directory"),
            start=config.get("start", 1),
        )

    def run(self):
        print("Starting Pipeline")
        process = psutil.Process()
        pipeline_start_perf = time.perf_counter()
        pipeline_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report.start_time = pipeline_start_time

        self.report.system_info = collect_system_info()

        #The first step will take from the input directory
        step_input_directory = self.input_directory

        for i, step in enumerate(self.steps, start=self.start):
            #Set the output directory for the step as {step_number}_{step_name}
            step_output_directory = os.path.join(self.output_directory, f"{i}_{step.name}")

            step_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            #The input directory of the next step will be output by the 
            step_input_directory = step.run(step_input_directory, step_output_directory)

            step_finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            #Get some stats from the step to fill the pipeline
            self.report.step_metadata[f"{i}_{step.name}"] = {
                "start_time":step_start_time,
                "finish_time":step_finish_time,
                "input_documents":step.report.input_documents,
                "output_documents":step.report.output_documents,
            }

            self.report.step_metadata[f"{i}_{step.name}"].update(step.report.step_stats)

            if i == 1:
                self.report.input_documents = step.report.input_documents
                self.report.input_files = step.report.input_files

            self.report.output_documents = step.report.output_documents

            self.report.output_directory = os.path.relpath(step_input_directory, self.output_directory)

        print("Finished Pipeline")
        
        pipeline_finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pipeline_runtime = time.perf_counter() - pipeline_start_perf
        self.report.finish_time = pipeline_finish_time

        self.report.execution_info = {
            "runtime_seconds": pipeline_runtime,

            "memory_rss_mb": round(
                process.memory_info().rss / (1024 ** 2),
                2
            ),

            "cpu_percent": process.cpu_percent(),
        }

        report_path = os.path.join(self.output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)

        print("Writing Pipeline Report")
        
        self.report.write_json(report_path)