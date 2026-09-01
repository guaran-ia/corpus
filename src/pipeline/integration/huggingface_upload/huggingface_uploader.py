import os
import json
from typing import Callable, Tuple
import yaml
import tempfile
from ..versioning.dataset_version import DatasetVersion
import shutil
from huggingface_hub import HfApi, login

#Functions that prepare data that is to be included in the dataset configs
DataPrepFunction = Callable[[str, str], Tuple[str, str]]

#Functions that prepare additional files
AdditionalPrepFunction = Callable[[str, str], None]

class HuggingFaceDatasetUploader():

    def __init__(self, repository:str, dataset_path:str, data_preparation_functions: list[DataPrepFunction], additional_functions:list[AdditionalPrepFunction] = None):
        self.data_preparation_functions = data_preparation_functions
        self.additional_functions = additional_functions
        self.dataset_path = dataset_path
        self.repository = repository

    @staticmethod
    def check_yaml_config(config_text:str) -> tuple[bool, list]:

        "Checks that a yaml config text is valid, according to huggingface"

        clean_text = config_text.strip().strip('---').strip()

        data_paths = []

        try:
            #Check that the config has a valid format
            data = yaml.safe_load(clean_text)

            #If the data succesfully loaded and it has configs
            if data and 'configs' in data:

                #Check each config
                for config in data['configs']:

                    #Check that the config has a name
                    config_name = config.get('config_name')

                    if not config_name:
                        print("Config has no name")
                        return False, []

                    #Check there are either data files or a data directory listed
                    if 'data_dir' not in config and 'data_files' not in config:
                        print(f"Files not specified for config {config_name}")
                        return False, []
                    
                    data_dir = config.get('data_dir', '')

                    data_files = config.get('data_files', [])
                    
                    #If there are any data files
                    if data_files:
                        #print(f"Found data files in config {config_name}")
                        #Data files is a single path
                        if type(data_files) == str:
                            full_path = os.path.join(data_dir, data_files)
                            data_paths.append(full_path)
                        #There are multiple splits
                        elif type(data_files) == list:
                            df_keys = data_files[0].keys()

                            for file in data_files:
                                
                                if file.keys() != df_keys:
                                    print(f"Mismatched keys for data files in config {config_name}")

                                    
                                path = file.get('path', [])

                                split = file.get('split', '')

                                if not path:
                                    print(f"No file paths found for split {split}")
                                    return False, []

                                if type(path) == str:
                                    full_path = os.path.join(data_dir, path)
                                    data_paths.append(full_path)
                                elif type(path) == list:
                                    for pth in path:
                                        full_path = os.path.join(data_dir, pth)
                                        data_paths.append(full_path)   
                        else:
                            print("Invalid data_files type")
                            return False, []
                    #If there are no data files and just a directory
                    else:
                        data_paths.append(f'./{data_dir}/')
                return True, data_paths
            else:
                print("No configs found")
                return False, []
        except yaml.YAMLError as exc:
            print(f"Invalid YAML formatting: {exc}")
            return False, []

    @staticmethod
    def _prepare_readme(readme_path:str, yaml_string:str) -> None:

        readme_content = ""

        #Read existing content if the file exists
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
                
        #Ensure there's a trailing newline so it doesn't merge with the first line
        if not yaml_string.endswith("\n"):
            yaml_string += "\n"
            
        #Write the combined text back to the file
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(yaml_string + readme_content)

    def upload(self, token, tag_message = None, **configs):
        #Upload files to the dataset

        #Check that the given path is a dataset path
        try:
            dataset_registry = os.path.join(self.dataset_path, "dataset.json")
            dataset_version = DatasetVersion.load_and_validate(dataset_registry)
        except:
            print("Directory is not a valid dataset version")
            return

        #Useful Files of the dataset
        data_path = os.path.join(self.dataset_path, dataset_version.files_path)

        #Temporary directory that is used as the repository root, the uploaded files will mirror the structure of this directory        
        with tempfile.TemporaryDirectory(suffix="_upload_temp", dir = ".") as temp_directory:

            yaml_config_full = []

            print(f"Created temporary directory at {temp_directory}")

            for data_prep_func in self.data_preparation_functions:
                try:
                    #Try to run the data preparation functions
                    yaml_configs, file_list = data_prep_func(data_path, temp_directory)
                    
                    #For data preparation functions that output configs
                    if yaml_configs:

                        #Check that the yaml configs are correct
                        valid_yaml, listed_files = self.check_yaml_config(yaml_configs)

                        if not valid_yaml:
                            print("Invalid YAML config found")
                            return
                        
                        #Check that all files listed in the YAML config exist in the directory (at least in theory)
                        files_exist = set(listed_files).issubset(set(file_list))

                        if not files_exist:
                            print("Some files in the configs were not found in the directory")
                            return

                        #Add the configuration to the list of configs
                        clean_str = yaml_configs.strip().strip('---').strip()
                        parsed = yaml.safe_load(clean_str)
                        
                        if parsed and "configs" in parsed:
                            yaml_config_full.extend(parsed["configs"])
                except Exception as e:
                    print(f"Failed to compile data files: {e}")
                    return

            merged_data = {"configs": yaml_config_full}
            merged_yaml_str = f"---\n{yaml.dump(merged_data, sort_keys=False).strip()}\n---"

            for additional_func in self.additional_functions:
                try:
                    additional_func(self.dataset_path, temp_directory)
                except Exception as e:
                    print(f"Failed to produce additional files: {e}")
                    return

            #Check if there is a README.md file in the temp repository or the dataset repository

            readme_path = os.path.join(temp_directory, "README.md")

            if os.path.isfile(readme_path):
                print("README found after preparation functions")
            else:
                print("No README found after preparation function")
                if os.path.isfile(os.path.join(self.dataset_path, "README.md")):
                    print("Found README in dataset directory, copying")
                    shutil.copy2(os.path.join(self.dataset_path, "README.md"), temp_directory)
                else:
                    print("No README files found in dataset directory, creating new file")
                    with open(readme_path, "w") as file:
                        pass

            print(readme_path)

            self._prepare_readme(readme_path=readme_path, yaml_string=merged_yaml_str)

            #Try to upload all files in the temporary directory
            try:
                print(f"Authenticating with Hugging Face Hub for repo: {self.repository}")
                login(token=token)

                api = HfApi()

                print("📤 Mirroring workspace to the Hub (wiping stale remote files)...")

                api.upload_folder(
                    folder_path=str(temp_directory),
                    repo_id=self.repository,
                    repo_type="dataset",
                    **configs
                )

                api.create_tag(
                    repo_id=self.repository,
                    repo_type="dataset",
                    tag=dataset_version.version_tag,
                    tag_message=tag_message if tag_message else ""
                )
                
            except Exception as e:
                print(f"Failed to upload the files to HuggingFace: {e}")


