from abc import ABC, abstractmethod

class PipelineStep(ABC):
    def __init__(self, input_directory:str, output_directory:str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    @abstractmethod
    def run(self):
        raise NotImplementedError
