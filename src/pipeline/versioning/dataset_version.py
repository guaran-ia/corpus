import os
from pydantic import BaseModel, Field, ValidationError
from ..pipeline_report import PipelineReport
import json

class DatasetVersion(BaseModel):
    version_tag: str = Field(description="The version tag for the dataset release. It should follow [Semantic Versioning](https://semver.org/) rules.")
    files_path: str = Field(description = "The relative path of the useful dataset files")
    changes: str = Field(description="The changes with respect to previous versions of the dataset, or an initial version description")
    previous_version: str = Field(description="Relative path to the previous dataset version registry", default_factory=str)

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_and_validate(cls, path: str) -> "DatasetVersion":
        with open(path, "r") as f:
            data = json.load(f)
        # This will raise a ValidationError if the JSON doesn't match the schema
        return cls.model_validate(data)

def register_release(directory, version, changes, previous_version = "") -> DatasetVersion:

    """Registers a directory as a dataset version. The directory should be an output directory of a Pipeline execution, whether with one or many steps.

    - **directory:** the relative directory with all the documents for the dataset.
    - **version:** the version number, according to [Semantic Versioning](https://semver.org/) rules.
    - **changes:** description of the changes 
    """
    try:
        report_path = os.path.join(directory, "report.json")
        pipeline_report = PipelineReport.load_and_validate(report_path)
    except:
        print("Found no Pipeline Report")
        return
    
    dataset_version = DatasetVersion(
        version_tag=version,
        files_path=pipeline_report.output_directory,
        changes = changes,
        previous_version=previous_version
    )

    output_path = os.path.join(directory, "dataset.json")
    dataset_version.write_json(output_path)

    



    