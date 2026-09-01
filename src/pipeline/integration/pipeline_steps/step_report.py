from pydantic import BaseModel, Field
from typing import Any
import json

class StepReport (BaseModel):
    step_name: str

    input_directory: str = Field(default_factory=str, description= "Relative directory where the input files for the step can be found")
    output_directory: str = Field(default_factory=str, description= "Relative directory where the output files for the step can be found")

    step_stats: dict = Field(default_factory=dict, description= "Any information or statistics that are specific to the step. For example, the number of removed documents.")      #Statistics specific to the step, including number of duplicates, or skipped files

    input_documents: int = 0        #Total Documents before pipeline
    output_documents: int = 0       #Remaining documents after the pipeline

    input_files: int = 0

    file_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)     #File-by-file breakdown of the step, if available

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_and_validate(cls, path: str) -> "StepReport":
        with open(path, "r") as f:
            data = json.load(f)
        # This will raise a ValidationError if the JSON doesn't match the schema
        return cls.model_validate(data)