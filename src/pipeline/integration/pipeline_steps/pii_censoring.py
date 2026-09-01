import os
from .pipeline_step import PipelineStep
from ..utils.loader import load_jsonl
from ..utils.writer import JSONLWriter
from .step_report import StepReport
from tqdm import tqdm
import warnings
import faker
import random


class PIICensoring(PipelineStep):
    def __init__(self, censoring_method:str = "tag", overlap_handle:dict = {"merge":True, "priority":"right"}):

        self.step_report = StepReport(step_name=self.name)

        self.step_report.step_stats = {
            "censoring_method":censoring_method,
            "documents_with_pii":0,
            "documents_with_pii_percentage":0.0,
            "pii_counts_by_type": {
                "email": 0,
                "phone": 0,
                "ip": 0
            }
        }
        self.censoring_method = censoring_method

        self.overlap_handle = overlap_handle

        self.fake = faker.Faker()

    @property
    def name(self):
        return "pii_censoring"

    @property
    def report(self) -> StepReport:
        return self.step_report

    def _handle_overlap(self, first_span:dict, second_span:dict):

        spans = []

        #Check order, ensure that the first span is always "on the left"
        if first_span["start"] >= second_span["start"]:
            first_span, second_span = second_span, first_span

        #For nested spans, we prioritise the biggest one
        if second_span["end"] <= first_span["end"]:
            spans.append(first_span)
        elif first_span['end'] <= second_span['end']:
            spans.append(second_span)
        #For partial overlaps
        else:
            #Calculate the overlap length
            overlap = first_span['end'] - second_span['start'] + 1

            #If both spans are of the same type, and merging has been selected
            if first_span['type'] == second_span['type'] and self.overlap_handle['merge']:
                merged_span = {
                    'type': first_span['type'],
                    'start': first_span['start'],
                    'end': second_span['end'],
                    'text': first_span['text'] + second_span['text'][overlap:]      #Cut the second span so that the text matches correctly
                }

                spans.append(merged_span)
            #If either the spans differ in type or we are not meant to merge same-type spans
            else:
                #We use the priority to define which span preserves it's whole text
                priority = self.overlap_handle['priority']

                #If priority is according to size, define whether the left or right span is prioretized
                if self.overlap_handle == "large":
                    priority = "left" if len(first_span['text']) > len(second_span['text']) else "right"
                elif self.overlap_handle == "small":
                    priority = "left" if len(first_span['text']) < len(second_span['text']) else "right"
            

                #If we prioritise the left span, cut the right span
                if priority == "left":
                    right_span = {
                        'type': second_span['type'],
                        'start': first_span['end'] + 1,
                        'end': second_span['end'],
                        'text': second_span['text'][overlap:]
                    }
                    spans.append(first_span)
                    spans.append(right_span)
                #If we prioritise the right span, cut the left span
                else:
                    left_span = {
                        'type':first_span['type'],
                        'start': first_span['start'],
                        'end': second_span['start'] - 1,
                        'text': first_span['text'][:-overlap]
                    }
                    spans.append(left_span)  
                    spans.append(second_span)

        return spans

    def _normalise_spans(self, spans: list[dict]):
        """Handles span overlap according to specifications"""

        if not spans:
            return []
        
        #Sort spans by starting point
        sorted_spans = sorted(spans, key=lambda x: x["start"])

        normalised_spans = []
        current_span = dict(sorted_spans[0])

        for next_span in sorted_spans[1:]:
            #If there is an overlap, handle it
            if next_span["start"] <= current_span["end"]:
                handled_spans = self._handle_overlap(current_span, next_span)
                if len(handled_spans) > 1:
                    normalised_spans.extend(handled_spans[:-1])
                current_span = handled_spans[-1]
            else:
                normalised_spans.append(current_span)
                current_span = dict(next_span)

        return normalised_spans

    def _censor(self, span:dict):
        """Replaces the span with the specified method"""
        if self.censoring_method == "tag":
            censor = f"{{{span['type']}}}"
        elif self.censoring_method == "stars":
            censor = "*" * len(span['text'])
        elif self.censoring_method == "replace":
            if span["type"] == 'email':
                censor = self.fake.email()
            elif span["type"] == "ip":
                censor = self.fake.ipv4()
            elif span["type"] == "phone":
                #Generate a fake phone number, but with a paraguayan prefix
                prefix = "+595"
                censor = prefix + str(random.randint(100_000_000, 999_999_999))

        return censor

    def run(self, source_directory:str, output_directory:str):
        print(f"Starting PII Censoring with {self.censoring_method}")

        #Create a directory for the censored files
        censored_directory = os.path.join(output_directory, "censored")

        #Write with all the metadata fields
        write_config = {"include_metadata":True, "metadata_fields":"*"}

        #Ensure that the directories exist
        os.makedirs(censored_directory, exist_ok = True)

        #Iterate over all files in the directory
        files = [f for f in os.listdir(source_directory) if f.endswith(".jsonl")]
        self.step_report.input_files = len(files)

        pbar_files = tqdm(files, desc="Processing files")

        for file in pbar_files:
            if file.endswith(".jsonl"):
                corpus_file_name = os.path.splitext(os.path.basename(file))[0]

                with JSONLWriter(os.path.join(censored_directory, file)) as censored_writer:

                    tqdm.write(f"Checking file {file}")

                    file_data = {
                        "input_documents":0,
                        "documents_with_pii":0,
                        "documents_with_pii_percentage":0.0,
                        "pii_counts_by_type": {
                            "email": 0,
                            "phone": 0,
                            "ip": 0
                        }
                    }

                    for document in load_jsonl(os.path.join(source_directory, file), field_map = {"text":"text", "id":"id"}, load_metadata = True, generate_id=False):
                        pbar_files.set_postfix({
                            "file": file,
                            "doc": document.id
                        })

                        self.step_report.input_documents +=1
                        file_data["input_documents"] +=1

                        try:
                            if document.metadata['has_pii']:

                                file_data["documents_with_pii"] += 1
                                self.step_report.step_stats['documents_with_pii'] += 1

                                #Normalise the spans to avoid potential overlapping issues
                                normalised_spans = self._normalise_spans(document.metadata['pii_spans'])

                                #Sort the spans in reverse order, to avoid issues with the length of the text when censoring
                                sorted_spans = sorted(normalised_spans, key=lambda x: x["start"], reverse=True)

                                for span in sorted_spans:
                                    #print("censoring")
                                    censor = self._censor(span)
                                    #print(print(censor))

                                    file_data["pii_counts_by_type"][span["type"]] += 1
                                    self.step_report.step_stats["pii_counts_by_type"][span["type"]] +=1

                                    start = span["start"]
                                    end = span["end"] + 1

                                    span_text = document.text[start:end]

                                    if span_text == span["text"]:
                                        censored_text = document.text[:start] + censor + document.text[end:]
                                        document.text = censored_text
                                    else:
                                        warnings.warn(f"Found mismatching span texts in document {document.id}: \n\nSpan Text:{span["text"]}\nDocument Text: {span_text}", UserWarning)
                                
                            censored_writer.write(document, **write_config)
                        except Exception as e:
                            print(f"Failed to evaluate document {document.id}: {e}")

                    file_data["documents_with_pii_percentage"] = file_data["documents_with_pii"]/file_data["input_documents"] 
                    self.step_report.file_stats[corpus_file_name] = file_data

        self.step_report.step_stats["documents_with_pii_percentage"] = self.step_report.step_stats["documents_with_pii"]/self.step_report.input_documents

        #Write final report
        print("Writing report")
        report_path = os.path.join(output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)

        self.step_report.output_directory = censored_directory

        self.step_report.write_json(report_path)

        return censored_directory


                



