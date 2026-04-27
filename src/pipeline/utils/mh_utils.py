import unicodedata
import re

from nltk.tokenize import TweetTokenizer

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


_tokenizer = TweetTokenizer(preserve_case=False)

def tokenize(text: str):
    """
    Tokenizes text using NLTK TweetTokenizer and yields alphabetic tokens only.
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token

def get_shingles(text: str, size: int = 5) -> set[str]:
    """
    Generate shingles (n-grams of tokens) from the input text.
    
    Args:
        text (str): The input text to generate shingles from.
        size (int): The number of tokens in each shingle (default is 5).
    
    Returns:
        set[str]: A set of shingles, where each shingle is a space-separated
                  sequence of SHINGLE_SIZE tokens. If the text contains fewer
                  tokens than SHINGLE_SIZE, returns a set containing a single
                  shingle with all tokens joined by spaces.
    """
    shingles = set()
    text_list = list(tokenize(text))
    if len(text_list) < size:
        return set([' '.join(text_list)])
    for i in range(len(text_list) - size + 1):
        shingle = ' '.join(text_list[i:i + size])
        shingles.add(shingle)
    return shingles