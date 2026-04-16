import os

from .url_deduplication import execute_url_deduplication


def main() -> None:
    """
    Run the URL deduplication pipeline using the project data and outputs
    directories.

    This runner resolves the project root relative to the current file,
    builds the paths to the data and outputs directories, and executes
    URL deduplication across all processed corpora.

    Returns:
        None
    """
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


if __name__ == "__main__":
    main()