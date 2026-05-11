from .pipeline_step import PipelineStep
from .pipeline_report import PipelineReport
import os
from datetime import datetime
import json

class Pipeline():
    def __init__(self, steps:list[PipelineStep], input_directory:str, output_directory:str):
        self.steps = steps
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.report = PipelineReport(total_steps=len(steps))

    def run(self):
        print("Starting Pipeline")
        pipeline_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report.start_time = pipeline_start_time

        #The first step will take from the input directory
        step_input_directory = self.input_directory

        for i, step in enumerate(self.steps, start=1):
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
                "remaining_documents":step.report.remaining_documents,
                "removed_documents":step.report.removed_documents
            }

            if i == 1:
                self.report.input_documents = step.report.input_documents
                self.report.input_files = step.report.input_files

            self.report.output_documents = step.report.remaining_documents
            self.report.removed_documents += step.report.removed_documents

        print("Finished Pipeline")
        
        pipeline_finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report.finish_time = pipeline_finish_time

        report_path = os.path.join(self.output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)

        print("Writing Pipeline Report")
        
        self.report.write_json(report_path)