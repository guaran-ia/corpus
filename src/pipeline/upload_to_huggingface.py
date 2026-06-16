import os
from .utils.loader import load_jsonl
from .utils.writer import JSONLWriter
from huggingface_hub import HfApi, login
import shutil

def create_temp_files(source_path:str, temp_compile_path:str):
    yaml_configs = []

    os.makedirs(temp_compile_path, exist_ok = True)

    variants = [
        "individual_full",
        "compiled_full",
        "individual_lite",
        "compiled_lite"
    ]

    for variant in variants:
        os.makedirs(os.path.join(temp_compile_path, variant), exist_ok=True)

    #Write Configs
    write_config_full = {"include_metadata":True, "metadata_fields":"*"}
    write_config_lite = {"include_metadata":False, "metadata_fields":None}

    #Writer for compiled JSON files
    compiled_full_writer = JSONLWriter(os.path.join(temp_compile_path, "compiled_full", "compiled_full.jsonl"))
    compiled_lite_writer = JSONLWriter(os.path.join(temp_compile_path, "compiled_lite", "compiled_lite.jsonl"))

    #YAML configs for compiled files
    yaml_configs.append("- config_name: compiled_full\n  data_files: compiled_full/compiled_full.jsonl")
    yaml_configs.append("- config_name: compiled_lite\n  data_files: compiled_lite/compiled_lite.jsonl")

    with compiled_full_writer, compiled_lite_writer:
    
        for file in os.listdir(source_path):

            corpus_name = os.path.splitext(file)[0]

            #Not in source files
            if not file.endswith(".jsonl") or corpus_name=='coreguapa':
                continue
            
            file_full_writer = JSONLWriter(os.path.join(temp_compile_path, "individual_full", f"{corpus_name}_full.jsonl"))
            file_lite_writer = JSONLWriter(os.path.join(temp_compile_path, "individual_lite", f"{corpus_name}_lite.jsonl"))

            yaml_configs.append(f"- config_name: {corpus_name}_full\n  data_files: individual_full/{corpus_name}_full.jsonl")
            yaml_configs.append(f"- config_name: {corpus_name}_lite\n  data_files: individual_lite/{corpus_name}_lite.jsonl")

            with file_full_writer, file_lite_writer:

                file_documents = load_jsonl(
                    os.path.join(source_path, file), 
                    field_map={"id":"id", "text":"text"}, 
                    load_metadata=True, 
                    metadata_fields="*"
                )

                for doc in file_documents:
                    #null values normalization
                    if doc.metadata and isinstance(doc.metadata, dict):
                        if 'duplicate' in doc.metadata.keys():
                            del doc.metadata['duplicate']
                        for key, val in doc.metadata.items():
                            if val is None:
                                doc.metadata[key] = ""

                    compiled_full_writer.write(doc, **write_config_full)
                    file_full_writer.write(doc, **write_config_full)
                    compiled_lite_writer.write(doc, **write_config_lite)
                    file_lite_writer.write(doc, **write_config_lite)
    
    config_str = "---\n" + "\n".join(yaml_configs) + "\n---"

    return config_str


      
def upload_to_hf_hub(token: str, repo_id: str, local_folder_path: str, version: str = None, commit_message:str = None):
    """Initializes the repository and pushes all staged folders in one batch."""
    print(f"🚀 Authenticating and verifying repository: {repo_id}")
    login(token=token)
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    print("📤 Uploading all variants in a single optimized push...")

    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message = commit_message,
        delete_patterns = "*",
    )

    # api

    # if version:
    #     api.create_tag(
    #         repo_id = repo_id,
    #         tag = version,
    #     )
    print("\n🎉 Everything uploaded successfully!")

def pipeline_processing_hugging_face_upload(pipeline_execution_directory: str):

    for dirpath, dirnames, filenames in os.walk(pipeline_execution_directory):

        for file in filenames:

            full_path = os.path.join(dirpath, file)

            print(f"  -> File: {full_path}")

def compile_discarded(pipeline, out_dir):
    os.makedirs(os.path.join(out_dir, "removed"), exist_ok=True)

    for ent in os.listdir(pipeline):
        if os.path.isdir(os.path.join(pipeline, ent)):
            source_dir = os.path.join(pipeline, ent, "remove")

            step_name = "_".join(str(ent).split("_")[1:])

            dest_dir = os.path.join(out_dir, "removed", step_name)

            os.makedirs(dest_dir, exist_ok=True)

            for file_name in os.listdir(source_dir):
                print(file_name)
                source_path = os.path.join(source_dir, file_name)
                dest_path = os.path.join(dest_dir, file_name)

                if os.path.isfile(source_path):
                    shutil.copy(source_path, dest_path)


def upload_discarded(token, repo_id, local_folder_path, commit_message):
    print(f"🚀 Authenticating and verifying repository: {repo_id}")
    login(token=token)
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    print("📤 Uploading all variants in a single optimized push...")

    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message = commit_message,
    )


if __name__ == "__main__":
    # source_path = "outputs/kuatia_1_0_0/2_minhash_deduplication/keep"
    # temp_path = "outputs/hf_upload_temp/"

    # yaml_config = create_temp_files(source_path=source_path, temp_compile_path=temp_path)

    tok = ""
    repo_id = "guaran-ia/kuatia"

    # version = "1.0.0"

    # upload_to_hf_hub(
    #     tok, 
    #     repo_id, 
    #     temp_path,
    #     version,
    #     commit_message="Corrected metadata fields"
    # )

    # with open("outputs/hf_upload_config.txt", "w", encoding="utf-8") as file:
    #     file.write(yaml_config)

    pipeline = "outputs/kuatia_1_0_0"
    temp_path = "outputs/hf_upload_temp_removed/"

    #compile_discarded(pipeline, temp_path)

    upload_discarded(
        tok,
        repo_id,
        temp_path,
        commit_message = "Uploaded discarded files from first version"
    )
