import os
from ..pipeline_step import PipelineStep
from ..utils.loader import load_jsonl
from ..utils.writer import JSONLWriter
from ..step_report import StepReport
from datetime import datetime
import shutil
import json
from tqdm import tqdm

class URLDeduplication(PipelineStep):
    def __init__(self, duplicate_ids_path:str, ignore_files_path:str,):
        self.duplicate_ids_path = duplicate_ids_path
        self.ignore_files_path = ignore_files_path
        self.step_report = StepReport(step_name=self.name)
        self.step_report.step_stats = {
            "processed_files":0,
            "processed_documents":0,
            "skipped_files":0,
            "skipped_documents":0,
            "duplicate_documents":0,
            "non_duplicate_documents":0,
            "unique_clusters":0
        }

    @property
    def name(self):
        return "url_deduplication"
    
    @property
    def report(self) -> StepReport:
        return self.step_report
    
    def run(self, source_directory:str, output_directory:str):
        print("Starting URL Deduplication")
        #Deduplication with timestamp
        #url_dedup_name = f"filter_url_{datetime.now().strftime("%Y%m%d%H%M%S")}"

        #Different directories for files to keep and remove
        keep_directory = os.path.join(output_directory, "keep")
        remove_directory = os.path.join(output_directory, "remove")

        write_config = {"include_metadata":True, "metadata_fields":"*"}

        os.makedirs(keep_directory, exist_ok = True)
        os.makedirs(remove_directory, exist_ok = True)

        #Load ignore files list
        with open(self.ignore_files_path, "r") as f:
            ignore_files = set(json.load(f))

        #Load duplicate ids map
        with open(self.duplicate_ids_path, "r") as f:
            dup_map = json.load(f)

        #Each document cluster has a representative, which is *not necessarily* the document that will be kept
        #This set will keep track of the representatives (clusters) which have already have a document written 
        seen_reps = set()

        #Iterate over all files in the directory
        files = [f for f in os.listdir(source_directory) if f.endswith(".jsonl")]
        self.step_report.input_files = len(files)

        pbar_files = tqdm(files, desc="Processing files")

        for file in pbar_files:
            
            if file.endswith(".jsonl"):
                corpus_file_name = os.path.splitext(os.path.basename(file))[0]

                #Files where the same url is used for many documents are ignored and copied as-is 
                if corpus_file_name in ignore_files:

                    #Quickly iterate over the file to count the number of documents
                    skipped_file_documents = sum(1 for _ in load_jsonl(os.path.join(source_directory, file), field_map = {"text":"text", "id":"id"}, load_metadata = False, generate_id=False))
                    
                    #Copy the whole file to the kept documents directory
                    copied_path = shutil.copy2(os.path.join(source_directory, file), keep_directory)
                    tqdm.write(f"Skipping file {file}")

                    #Update report stats
                    self.step_report.step_stats["skipped_files"] +=1
                    self.step_report.step_stats["skipped_documents"] += skipped_file_documents
                    self.step_report.input_documents += skipped_file_documents
                    self.step_report.remaining_documents += skipped_file_documents

                    continue
                
                self.step_report.step_stats["processed_files"] +=1

                #Other files
                with JSONLWriter(os.path.join(keep_directory, file)) as keep_writer, \
                JSONLWriter(os.path.join(remove_directory, file)) as remove_writer:
                    
                    tqdm.write(f"Checking file {file}")
                    
                    file_data = {
                        "input_documents":0,
                        "remaining_documents":0,
                        "removed_documents":0,
                        "duplicate_documents":0,
                        "non_duplicate_documents":0
                    }

                    for document in load_jsonl(os.path.join(source_directory, file), field_map = {"text":"text", "id":"id"}, load_metadata = True, generate_id=False):
                        pbar_files.set_postfix({
                            "file": file,
                            "doc": document.id
                        })

                        self.step_report.input_documents +=1
                        self.step_report.step_stats["processed_documents"] +=1
                        file_data["input_documents"] +=1
        
                        try:
                            duplicates = dup_map.get(document.id, None)
                            #If the document id is present in the map keys, then it has at least one duplicate, so we conduct the proper checks
                            if duplicates:
                                
                                self.step_report.step_stats["duplicate_documents"] +=1
                                file_data["duplicate_documents"] +=1

                                #Get the document cluster
                                #For each document id, we can get all the other documents which have the same url, so a cluster consist of that plus the document id
                                #This calculation results in a list with the same items for all documents in the cluster, regardless of the id being mapped
                                cluster = [document.id] + duplicates

                                #Because the generated list is the same for all ids in the cluster, the min value will also be the same, so it is chosen as a representative
                                representative = min(cluster)

                                #If we haven't seen the representative, it means that no document from this cluster has been included yet, so we keep it
                                if representative not in seen_reps:

                                    self.step_report.remaining_documents +=1
                                    file_data["remaining_documents"] +=1

                                    keep_writer.write(document, **write_config)
                                    seen_reps.add(representative)
                                #If we have seen it, then another document from this cluster has been written before, so we remove it
                                else:
                                    self.step_report.removed_documents +=1
                                    file_data["removed_documents"] +=1

                                    remove_writer.write(document, **write_config)
                            #If it is not in the map keys, then it has no duplicates, so we keep it
                            else:
                                
                                self.step_report.step_stats["non_duplicate_documents"] +=1
                                file_data["non_duplicate_documents"] +=1

                                self.step_report.remaining_documents +=1
                                file_data["remaining_documents"] +=1
                            
                                keep_writer.write(document, **write_config)
                        except Exception as e:
                            print(e)
                            print(f"Doc ID: {document.id}")
                    
                    self.step_report.file_stats[corpus_file_name] = file_data

        self.step_report.step_stats["unique_clusters"] = len(seen_reps)

        #Write final report
        print("Writing report")
        report_path = os.path.join(output_directory, "report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok = True)

        self.step_report.write_json(report_path)

        return keep_directory


# if __name__ == "__main__":
#     url_deduplication = URLDeduplication(
#         duplicate_ids_path = "outputs/deduplication/url_202605022046/duplicate_ids.json",
#         ignore_files_path="outputs/deduplication/url_dedup_ignore_corpora.json",
#     )
    
#     keep_directory = url_deduplication.run(
#         source_directory = "data/processed",
#         output_directory = "outputs/filtering"
#     )

#     print(f"URL Deduplication completed, keep files saved to {keep_directory}")