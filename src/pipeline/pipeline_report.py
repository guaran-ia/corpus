from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime
import json


@dataclass
class PipelineReport:
    start_time: str | None = None
    finish_time: str | None = None

    total_steps: int = 0

    input_documents: int = 0         # Total documents before pipeline execution
    output_documents: int = 0        # Remaining documents after pipeline execution
    removed_documents: int = 0       # Total documents removed during pipeline execution

    input_files: int = 0

    #pipeline_stats: dict[str, Any] = field(default_factory=dict)
    step_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=2
            )