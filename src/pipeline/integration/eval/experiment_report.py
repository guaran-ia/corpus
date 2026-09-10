from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
import json

class ExperimentReport(BaseModel):
    name: str                       #The name of the experiment

    input_directory: str = Field(default_factory = str)
    output_directory:str = Field(default_factory = str)

    step_reports: dict[str, Any] = Field(default_factory = dict)

    pipeline_report: str = Field(default_factory=str)

    dataset_metrics: dict[str, Any] = Field(default_factory=dict)

    lm_eval_results: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    
