import os
from pydantic import BaseModel, Field, ValidationError
from .. import PipelineReport, StepReport
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

def register_release(directory:str, version:str, changes, previous_version = "") -> DatasetVersion:
    """Registers a directory as a Dataset version by adding a `dataset.json` file to the directory.

    Args:
        directory (str): The directory to register as a version. Ideally, it should be the output directory of a pipeline run, but it may also be the output directory of a single step run, or simply a collection of files.
    
    Returns:
        dataset_version (DatasetVersion): The corresponding DatasetVersion object.
    """

    report_path = os.path.join(directory, "report.json")
    report = None
    try:
        #Try to find a pipeline report
        report = PipelineReport.load_and_validate(report_path)
    except ValidationError:
        try:
            report = StepReport.load_and_validate(report_path)
        except:
            print("Found no reports, registering directory as a DatasetVersion")
    
    dataset_version = DatasetVersion(
        version_tag=version,
        files_path=report.output_directory if report else ".",
        changes = changes,
        previous_version=previous_version
    )

    output_path = os.path.join(directory, "dataset.json")
    dataset_version.write_json(output_path)

    



    