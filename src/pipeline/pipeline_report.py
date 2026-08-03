from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
import json

class PipelineReport(BaseModel):
    start_time: str | None = None
    finish_time: str | None = None

    total_steps: int = 0

    input_directory: str = Field(default_factory = str)
    output_directory:str = Field(default_factory = str)

    input_documents: int = 0         # Total documents before pipeline execution
    output_documents: int = 0        # Remaining documents after pipeline execution

    input_files: int = 0

    system_info: dict[str, Any] = Field(default_factory=dict)

    execution_info: dict[str, Any] = Field(default_factory=dict)

    step_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_and_validate(cls, path: str) -> "PipelineReport":
        with open(path, "r") as f:
            data = json.load(f)
        # This will raise a ValidationError if the JSON doesn't match the schema
        return cls.model_validate(data)