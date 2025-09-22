# Language Identifier Tool

This tool identifies the language of a given text using three different language 
identification models: [GlotLID](https://github.com/cisnlp/GlotLID), [FastText](https://huggingface.co/facebook/fasttext-language-identification), and [OpenLID](https://github.com/laurieburchell/open-lid-dataset). 
It provides a unified interface to these models, allowing for easy language detection 
and comparison of results.

The tool can identify up to k languages (k is passed as a parameter). If `k == 1`, majority voting is conducted to select the language chosen by at least two of the three models (if all three are available). The `GlotLID` result is preferred if it is among the agreeing voters; otherwise, the `OpenLID` result is used. For `k > 1`, the `GlotLID` prediction is returned.

:information_source: This tool is intended to be used by all language identification efforts throughout the Guarania project.

## Installation
1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd language_identifier/src/pipeline/language_identifier
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install OpenLID model into the models directory
    ```bash
    mkdir models
    cd models
    wget https://data.statmt.org/lid/lid201-model.bin.gz
    pigz -d lid201-model.bin.gz
    ```

## Usage
1.  Import the `LanguageIdentifier` class:
    ```python
    from language_identifier import LanguageIdentifier
    ```
2.  Initialize the `LanguageIdentifier` with the desired models (all enabled by default):
    ```python
    identifier = LanguageIdentifier(glotlid=True, fasttext=True, openlid=True)
    ```
3.  Identify the language of a text:
    ```python
    text = "Añetehápe, heta jevy upe ñe'ẽjoapy mokõiha reko hasy oiko haĝua térã ndaikatúi voi, ha upéare ijuky"
    result = identifier.identify_languages(text, k=3, raw_output=False)
    ```
    *   `text`: The text to identify the language of.
    *   `k`: The number of top languages to return for each model (default: 3).
    *   `raw_output`: Whether to return the raw output from each model (default: False).

    The `identify_languages` method returns a dictionary containing the language identification results. 
    If `raw_output` is `False`, the dictionary contains the following keys:
    *   `languages`: A list of (language code, confidence score) tuples, representing the distribution of predicted languages.
    *   `source`: The source of the prediction (e.g., 'glotlid', 'fasttext', 'openlid').
    *   `voting`: A string indicating how the final prediction was determined 
    *   (e.g., 'all\_agree', 'agree\_glotlib\_fasttext', 'inconclusive').
    
    Output example for `k=1`
    ```python
    {
        'languages': ('grn', 0.7193270921707153),
        'source': 'glotlid',
        'voting': 'all_agree'
    }
    ```

    Output example for `k>1`
    ```python
    {
        'languages': [('grn', 0.7193270921707153), ('spa', 0.2611099183559418)],
        'source': 'glotlid',
        'voting': 'not_applicable_k_greater_than_1'
    }
    ```

    If `raw_output` is `True`, the dictionary contains the raw output from each model, 
    with the keys 'glotlid', 'fasttext', and 'openlid'.

    :warning: Language Identifier models, except for `OpenLID`, are downloaded 
    and installed the first time they are used, which might result in a long execution time.

## Dependencies
*   Python 3.12+
*   Language identifiers: [GlotLID](https://github.com/cisnlp/GlotLID), [FastText](https://huggingface.co/facebook/fasttext-language-identification), [OpenLID](https://github.com/laurieburchell/open-lid-dataset)

The rest of dependencies are listed in the `requirements.txt` file.