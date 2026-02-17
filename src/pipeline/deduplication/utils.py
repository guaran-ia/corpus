import os
import re
import unicodedata


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