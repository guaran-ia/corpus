import json
import os
import re
import unicodedata

from tqdm import tqdm


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
                        

def canonicalize_text(text: str, replace_separators: bool = True, 
                      remove_punctuation: bool = True, convert_case: bool = True) -> str:
    """
    Canonicalize the input text by converting it to lowercase and removing extra whitespace.

    Args:
        text (str): The input text to be normalized.
    """
    
    WHITESPACE_RE = re.compile(r'\s+')
    CANONICAL_TRANSLATIONS = str.maketrans(
        {
            "\u2019": "'",
            "\u2018": "'",
            "\u02bc": "'",
            "\uff07": "'",
            "`": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u201f": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2010": "-",
            "\u2212": "-",
            "\u2026": "...",
        }
    )
    
    # Unicode normalization: separate base characters and diacritics
    can_text = unicodedata.normalize('NFKD', text)
    # Remove diacritics (acute accents and nasal tildes)
    can_text = ''.join(
        ch for ch in text
        if unicodedata.category(ch) != 'Mn'
    )
    # Apply a precomputed translation mapping to normalize apostrophes and dashes
    can_text = can_text.translate(CANONICAL_TRANSLATIONS)
    # Replace all Unicode separator characters (unicode category that starts with Z) 
    # with a single space
    if replace_separators:
        can_text = ''.join(
            ' ' 
            if unicodedata.category(char).startswith('Z') else char for char in can_text
        )
    # Remove punctuation characters (commas, periods, dashes, quotes, apostrophes, etc.)
    if remove_punctuation:
        can_text = ''.join(
            char for char in can_text
            if not unicodedata.category(char).startswith('P')
        )
    # Convert to lowercase
    if convert_case:
        can_text = can_text.lower()
    # Collpase multiple whitespace characters into a single space
    can_text = WHITESPACE_RE.sub(' ', can_text)
    # Remove leading and trailing whitespace
    can_text = can_text.strip()

    return can_text


def add_id_corpus(corpus: list, corpus_name: str) -> list:
    """
    Add an 'id' field to each document in the corpus, using the corpus name and document index.

    Args:
        corpus (list): A list of documents (dictionaries) in the corpus.
        corpus_name (str): The name of the corpus, used as a prefix for the ID.

    Returns:
        list: The input corpus with an added 'id' field for each document.
    """
    for idx, doc in enumerate(corpus):
        doc['id'] = f'{corpus_name}_{idx}'
    return corpus


def add_id_corpora(data_dir):
    """
    Add an 'id' field to each document in all corpora located in the data directory.

    Args:
        data_dir (str): The directory containing the corpora subdirectories.
    """
    corpora_dir = os.path.join(data_dir, 'processed')
    corpus_files = [f for f in os.listdir(corpora_dir) if os.path.isfile(os.path.join(corpora_dir, f)) and f.endswith('.jsonl')]
    for corpus_file in tqdm(corpus_files, desc='Adding IDs to corpora'):
        corpus_name = os.path.splitext(corpus_file)[0]
        corpus_path = os.path.join(corpora_dir, corpus_file)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [json.loads(line) for line in f]
        updated_corpus = add_id_corpus(corpus, corpus_name)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for doc in updated_corpus:
                f.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, 'data')
    add_id_corpora(data_dir)