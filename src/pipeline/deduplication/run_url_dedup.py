import os

from .fix_has_duplicate_url import execute_url_deduplication


if __name__ == "__main__":
    project_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(project_dir, "outputs")

    execute_url_deduplication(data_dir, output_dir)