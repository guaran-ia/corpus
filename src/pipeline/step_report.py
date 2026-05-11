from dataclasses import dataclass, field, asdict
from typing import Any
import json

@dataclass
class StepReport:
    step_name: str

    step_stats: dict = field(default_factory=dict)      #Statistics specific to the step, including number of duplicates, or skipped files

    input_documents: int = 0        #Total Documents before pipeline
    remaining_documents: int = 0       #Remaining documents after the pipeline
    removed_documents: int = 0      #Documents removed in the pipeline

    input_files: int = 0

    file_stats: dict[str, dict[str, Any]] = field(default_factory=dict)     #File-by-file breakdown of the step, if available

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path:str) -> None:

        with open(path, "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=2
            )