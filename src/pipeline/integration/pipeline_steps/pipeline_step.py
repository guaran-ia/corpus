from abc import ABC, abstractmethod
from .step_report import StepReport
from typing import Dict, Any
import yaml

class PipelineStep(ABC):
    @abstractmethod
    def run(self, input_directory:str, output_directory:str):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def report(self) -> StepReport:
        pass

    @classmethod
    def from_yaml(cls, yaml_path:str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @classmethod
    @abstractmethod
    def from_dict(cls, dict_config:Dict[str, Any]):
        pass
