from nltk.tokenize import TweetTokenizer

_tokenizer = TweetTokenizer(preserve_case=False)

def tokenize(text: str):
    """
    Tokeniza texto usando TweetTokenizer.
    Devuelve solo tokens alfabéticos (unicode).
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token

