from .pipeline_step import PipelineStep
from .step_report import StepReport
import os
from tqdm import tqdm
from ..utils.writer import JSONLWriter
from ..utils.loader import load_jsonl

class HeuristicFilter(PipelineStep):
    """Filters documents using their metadata."""

    def __init__(self, filtering_condition:str):
        self.step_report = StepReport(step_name=self.name)
        self.step_report.step_stats = {
            "condition":filtering_condition,
        }
        self.filtering_condition = filtering_condition
        self._condition_compiled = compile(filtering_condition, "<heuristic_filter>", "eval")

    @property
    def name(self):
        return "heuristic_filter"

    @property
    def report(self) -> StepReport:
        return self.step_report

    def run(self, source_directory:str, output_directory:str):
        print(f"Starting Heuristic Filtering with condition {self.filtering_condition}")

        #Different directories for files to keep and remove
        keep_directory = os.path.join(output_directory, "keep")
        remove_directory = os.path.join(output_directory, "remove")

        #Write with all the metadata fields
        write_config = {"include_metadata":True, "metadata_fields":"*"}

        #Ensure that the keep and remove directories exist
        os.makedirs(keep_directory, exist_ok = True)
        os.makedirs(remove_directory, exist_ok = True)

        #Iterate over all files in the directory
        files = [f for f in os.listdir(source_directory) if f.endswith(".jsonl")]
        self.step_report.input_files = len(files)

        pbar_files = tqdm(files, desc="Processing files")

        for file in pbar_files:
            
            if file.endswith(".jsonl"):
                corpus_file_name = os.path.splitext(os.path.basename(file))[0]

                self.step_report.input_files +=1

                with JSONLWriter(os.path.join(keep_directory, file)) as keep_writer, \
                JSONLWriter(os.path.join(remove_directory, file)) as remove_writer:
                    
                    tqdm.write(f"Checking file {file}")

                    file_data = {
                        "input_documents":0,
                        "remaining_documents":0,
                        "removed_documents":0
                    }

                    #Read the file
                    for document in load_jsonl(os.path.join(source_directory, file), field_map = {"text":"text", "id":"id"}, load_metadata = True, generate_id=False):
                
                        pbar_files.set_postfix({
                            "file": file,
                            "doc": document.id
                        })

                        file_data["input_documents"] +=1
                        self.step_report.input_documents +=1

                        #Evaluate the condition
                        try:
                            match = bool(eval(self._condition_compiled, {"__builtins__": {}}, document.metadata))
                        except Exception as e:
                            print(f"Failed to evaluate document {document.id}: {e}")

                            #Don't filter the document
                            match = False
                        
                        #If the document matches the condition, write it to the removed directory
                        if match:
                            file_data['removed_documents'] += 1
                            self.step_report.removed_documents += 1
                            remove_writer.write(document, **write_config)
                        else:
                            file_data['remaining_documents'] +=1
                            self.step_report.remaining_documents += 1
                            keep_writer.write(document, **write_config)

                    self.step_report.file_stats[corpus_file_name] = file_data

        #Write final report
        print("Writing report")
        report_path = os.path.join(output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)

        self.step_report.write_json(report_path)

        return keep_directory

                    