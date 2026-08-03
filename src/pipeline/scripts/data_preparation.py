import os
from typing import Tuple
from ..utils.loader import load_jsonl
from ..utils.writer import JSONLWriter
import yaml
from tqdm import tqdm

def prepare_individual_lite(input_directory, output_directory) -> Tuple[str, str]:
    yaml_config = {"configs":[]}
    file_list = []

    subdirectory = os.path.join(output_directory, "individual_lite")
    os.makedirs(subdirectory, exist_ok= True)

    write_config = {"include_metadata":False, "metadata_fields":None}

    files = [f for f in os.listdir(input_directory)]
    pbar_files = tqdm(files, desc="Preparing Individual Lite Files")

    for file in pbar_files:

        corpus_name = os.path.splitext(file)[0]

        if not file.endswith(".jsonl") or corpus_name=='coreguapa':
            continue

        pbar_files.set_postfix({"corpus_file":file})

        config_name = f"{corpus_name}_lite"

        file_path = os.path.join(subdirectory, f"{corpus_name}_lite.jsonl") 

        config = {"config_name":config_name, "data_files":os.path.relpath(file_path, output_directory)}

        file_list.append(os.path.relpath(file_path, output_directory))

        with JSONLWriter(file_path) as writer:

            file_documents = load_jsonl(
                os.path.join(input_directory, file), 
                field_map={"id":"id", "text":"text"}, 
                load_metadata=False)

            for doc in file_documents:
                writer.write(doc, **write_config)

        yaml_config["configs"].append(config)

    yaml_config_string = yaml.safe_dump(yaml_config, default_flow_style=False)

    return yaml_config_string, file_list


def prepare_individual_full(input_directory, output_directory) -> Tuple[str, str]:
    yaml_config = {"configs":[]}
    file_list = []

    subdirectory = os.path.join(output_directory, "individual_full")
    os.makedirs(subdirectory, exist_ok=True)

    write_config = {"include_metadata":True, "metadata_fields":"*"}

    files = [f for f in os.listdir(input_directory)]
    pbar_files = tqdm(files, desc="Preparing Individual Full Files")

    for file in pbar_files:

        corpus_name = os.path.splitext(file)[0]

        if not file.endswith(".jsonl") or corpus_name=='coreguapa':
            continue

        pbar_files.set_postfix({"corpus_file":file})

        config_name = f"{corpus_name}_full"

        file_path = os.path.join(subdirectory, f"{corpus_name}_full.jsonl") 

        config = {"config_name":config_name, "data_files":os.path.relpath(file_path, output_directory)}

        file_list.append(os.path.relpath(file_path, output_directory))

        with JSONLWriter(file_path) as writer:

            file_documents = load_jsonl(
                os.path.join(input_directory, file), 
                field_map={"id":"id", "text":"text"}, 
                load_metadata=True, metadata_fields="*")

            for doc in file_documents:

                if doc.metadata and isinstance(doc.metadata, dict):
                    if 'duplicate' in doc.metadata.keys():
                        del doc.metadata['duplicate']
                    for key, val in doc.metadata.items():
                        if val is None:
                            doc.metadata[key] = ""

                writer.write(doc, **write_config)

        yaml_config["configs"].append(config)

    yaml_config_string = yaml.safe_dump(yaml_config, default_flow_style=False)

    return yaml_config_string, file_list


def prepare_compiled_full(input_directory, output_directory) -> Tuple[str, str]:
    yaml_config = {"configs":[]}
    file_list = []

    subdirectory = os.path.join(output_directory, "compiled_full")
    os.makedirs(subdirectory, exist_ok=True)

    write_config = {"include_metadata":True, "metadata_fields":"*"}

    file_path = os.path.join(subdirectory, f"compiled_full.jsonl")

    files = [f for f in os.listdir(input_directory)]
    pbar_files = tqdm(files, desc="Preparing Compiled Full File")

    with JSONLWriter(file_path) as writer:

        for file in pbar_files:

            corpus_name = os.path.splitext(file)[0]

            if not file.endswith(".jsonl") or corpus_name=='coreguapa':
                continue

            pbar_files.set_postfix({"corpus_file":file})

            file_documents = load_jsonl(
                os.path.join(input_directory, file), 
                field_map={"id":"id", "text":"text"}, 
                load_metadata=True, 
                metadata_fields="*"
            )

            for doc in file_documents:

                if doc.metadata and isinstance(doc.metadata, dict):
                    if 'duplicate' in doc.metadata.keys():
                        del doc.metadata['duplicate']
                    for key, val in doc.metadata.items():
                        if val is None:
                            doc.metadata[key] = ""

                writer.write(doc, **write_config)

        yaml_config['configs'].append({'config_name':'compiled_full', 'data_files':os.path.relpath(file_path, output_directory)})

        file_list.append(os.path.relpath(file_path, output_directory))

        yaml_config_string = yaml.safe_dump(yaml_config, default_flow_style=False)

    return yaml_config_string, file_list


def prepare_compiled_lite(input_directory, output_directory) -> Tuple[str, str]:
    yaml_config = {"configs":[]}
    file_list = []

    subdirectory = os.path.join(output_directory, "compiled_lite")
    os.makedirs(subdirectory, exist_ok=True)

    write_config = {"include_metadata":False, "metadata_fields":None}

    file_path = os.path.join(subdirectory, f"compiled_lite.jsonl")

    files = [f for f in os.listdir(input_directory)]
    pbar_files = tqdm(files, desc="Preparing Compiled Lite File")

    with JSONLWriter(file_path) as writer:

        for file in pbar_files:

            corpus_name = os.path.splitext(file)[0]

            if not file.endswith(".jsonl") or corpus_name=='coreguapa':
                continue

            pbar_files.set_postfix({"corpus_file":file})

            file_documents = load_jsonl(
                os.path.join(input_directory, file), 
                field_map={"id":"id", "text":"text"}, 
                load_metadata=False, 
            )

            for doc in file_documents:
                writer.write(doc, **write_config)

        yaml_config['configs'].append({'config_name':'compiled_lite', 'data_files':os.path.relpath(file_path, output_directory)})

        file_list.append(os.path.relpath(file_path, output_directory))

        yaml_config_string = yaml.safe_dump(yaml_config, default_flow_style=False)

    return yaml_config_string, file_list