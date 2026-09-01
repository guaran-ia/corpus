from .pipeline_step import PipelineStep
from .step_report import StepReport
import os
from tqdm import tqdm
from ..utils.writer import JSONLWriter
from ..utils.loader import load_jsonl
import re
import numpy as np

class HeuristicFilter(PipelineStep):
    """Filters documents using their metadata."""

    #Used to compile 
    PERCENTILE_PATTERN = re.compile(
        r'\b[pP](\d+(?:[._]\d+)?)\_([a-zA-Z_][a-zA-Z0-9_]*)\b'
    )

    def __init__(self, filtering_condition:str):
        self.step_report = StepReport(step_name=self.name)

        self.step_report.step_stats = {
            "condition":filtering_condition,
            "removed_documents":0,
        }

        self.filtering_condition = filtering_condition

        #Ensure valid filtering condition
        self._condition_compiled = compile(filtering_condition, "<heuristic_filter>", "eval")

        #Check for filtering conditions requiring percentiles
        self._percentile_requests = self._parse_percentile_requests()

    def _parse_percentile_requests(self) -> list[tuple[str, str, float]]:

        matches = self.PERCENTILE_PATTERN.findall(self.filtering_condition)

        requests = []

        for p_str, field in matches:
            
            normalized_p = p_str.replace('_', '.')
            p_val = float(normalized_p)
            
            var_name = f"p{p_str}_{field}"
            requests.append((var_name, field, p_val))
        return requests

    @property
    def name(self):
        return "heuristic_filter"

    @property
    def report(self) -> StepReport:
        return self.step_report

    def _collect_percentiles(self, files: list[str], source_dir: str) -> dict[str, float]:

        #Collect all values of the required fields only
        field_values = {field: [] for _, field, _ in self._percentile_requests}

        for file in tqdm(files, desc="Computing global percentiles"):
            file_path = os.path.join(source_dir, file)
            for document in load_jsonl(file_path, field_map={"id": "id"}, load_metadata=True, generate_id=False):
                meta = document.metadata
                for _, field, _ in self._percentile_requests:
                    val = meta.get(field)
                    if val is not None and isinstance(val, (int, float)):
                        field_values[field].append(val)

        #Compute the percentile values for each variable
        computed_vars = {}
        for var_name, field, p_val in self._percentile_requests:
            vals = field_values[field]
            if not vals:
                raise ValueError(f"No numeric data collected for metadata field '{field}' to compute {var_name}.")
            computed_threshold = float(np.percentile(vals, p_val))
            computed_vars[var_name] = computed_threshold

        self.step_report.step_stats['percentile_variable_values'] = computed_vars

        return computed_vars

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

        #Calculate any percentile thresholds required
        percentile_context = {}
        if self._percentile_requests:
            percentile_context = self._collect_percentiles(files, source_directory)

        pbar_files = tqdm(files, desc="Processing files")

        for file in pbar_files:
            
            if file.endswith(".jsonl"):
                corpus_file_name = os.path.splitext(os.path.basename(file))[0]

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

                        eval_scope = {**document.metadata, **percentile_context}

                        #Evaluate the condition
                        try:
                            match = bool(eval(self._condition_compiled, {"__builtins__": {}}, eval_scope))
                        except Exception as e:
                            print(f"Failed to evaluate document {document.id}: {e}")

                            #Don't filter the document
                            match = False
                        
                        #If the document matches the condition, write it to the removed directory
                        if match:
                            file_data['removed_documents'] += 1
                            self.step_report.step_stats["removed_documents"] += 1
                            remove_writer.write(document, **write_config)
                        else:
                            file_data['remaining_documents'] +=1
                            self.step_report.output_documents += 1
                            keep_writer.write(document, **write_config)

                    self.step_report.file_stats[corpus_file_name] = file_data

        #Write final report
        print("Writing report")
        report_path = os.path.join(output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)
        
        self.step_report.output_directory = os.path.relpath(keep_directory, output_directory)

        self.step_report.write_json(report_path)

        return keep_directory

                    