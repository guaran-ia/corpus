from nltk.tokenize import TweetTokenizer

_tokenizer = TweetTokenizer(preserve_case=False)

def tokenize(text: str):
    """
    Tokenizes text using NLTK TweetTokenizer and yields alphabetic tokens only.
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token

