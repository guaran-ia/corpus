from ..pipeline import Pipeline
from ..pipeline_steps.pii_censoring import PIICensoring
from ..versioning.dataset_version import DatasetVersion, register_release
import os

if __name__ == "__main__":
    previous_version = "kuatia/1_0_0"
    output_directory = "kuatia/1_1_0"

    dataset_registry = os.path.join(previous_version, "dataset.json")


    try:
        dataset_version = DatasetVersion.load_and_validate(dataset_registry)
    except:
        "Found no valid registry previous dataset version"
      
    input_directory = os.path.join(previous_version, dataset_version.files_path)

    print(input_directory)

    pii_censoring = PIICensoring(censoring_method="replace")

    pipeline = Pipeline(steps = [pii_censoring], 
                        input_directory=input_directory, 
                        output_directory=output_directory
                )

    pipeline.run()

    register_release(
        output_directory, 
        version="1.1.0", 
        changes="Added Censoring of Personal and Identifiable Information (PII)",
        previous_version=os.path.relpath(dataset_registry, output_directory))
    

