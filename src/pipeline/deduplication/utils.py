import os


def create_raw_corpora(corpus_dir: str, output_dir: str) -> None:
    """
    Create corpora file that aggregates the content of all corpus stored in corpus_dir. 
    The corpora file is save into the output directory.

    Args:
        corpus_dir (str): The directory containing the input corpora.
        output_dir (str): The directory where the raw corpora will be saved.
    """
    print(f'Creating corpora from {corpus_dir} and saving to {output_dir}...')
    corpus_files = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]
    output_path = os.path.join(output_dir, 'corpora.jsonl')
    with open(output_path, 'a', encoding='utf-8') as outfile:
        for corpus_file in corpus_files:
            corpus_path = os.path.join(corpus_dir, corpus_file)
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Process the line if necessary (e.g., remove extra whitespace)
                    processed_line = line.strip()
                    if processed_line:  # Only write non-empty lines
                        outfile.write(processed_line + '\n')