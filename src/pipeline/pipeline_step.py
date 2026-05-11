from abc import ABC, abstractmethod
from .step_report import StepReport

class PipelineStep(ABC):
    @abstractmethod
    def run(self, input_directory:str, output_directory:str):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def report(self) -> StepReport:
        return self.report
