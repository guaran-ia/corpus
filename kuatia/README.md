# KuatIA 📚

KuatIA (Guarani: document, book) is a dataset which compiles and curates Guarani documents from 43 sources as part of the GuaranIA project (No. ATN/OC-21347-PR), co-financed by the Inter-American Development Bank. It contains documents from diverse origins: synthetic and non-sinthetic, online and offline.

## Source Data 📥

KuatIA consists on 41 previously published corpora, as well as 2 corpora that have been obtained and/or digitalised by us. Below is a table with details about the corpora that make up KuatIA, with links to the original publications, when available

|Corpus Name|Authorship|Documents Used|
|:---:|:---:|:---:|

## Data Processing ⚙️

### Formatting and Data Fields 📑

Upon obtention, all documents from all sources were processed to match a unified **`JSON`** format. 

Each document consists of the following fields:

```python
{
    'text': '',  # corpus text
    'corpus': '', # corpus name
    'corpus_file': '', # corpus file name
    'source': '', # text source (if available)
    'url': '', # text source url (if available)
    'language': 'gnr', # iso-6393 code for Guarani
    'language_score': 0.0, # proportion of Guarani in the text
    'language_script': 'Latn', # Guarani script
    'language_score_source': '', # source of Guarani score
    'language_identification_method': '', # method used to identify language
    'num_words_split': 0, # number of words in text based on white-space split
    'num_words_punct_spacy': 0, # number of words in text based on the Spacy generic segmentator
    'num_words_no_punct_spacy': 0, # number of words in text based on the Spacy generic segmentator (excluding punctuation)
    'num_chars': 0 # number of characters in the text
}
```

To read more about our formatting processes, we refer you to our repositories:

- [**`🔗 Existing Guarani Corpora`**](https://github.com/guaran-ia/existing-guarani-corpora), which processes 39 open guarani corpora 
- [**`🔗 Oremba'e Exploration`**](https://github.com/guaran-ia/orembae-exploration), our digitalization and processing of the book "Che ñe'e, che purahéi"
- HLTK
- GuaraScrapper
- Something else

### Curation Pipeline 🗂️

To ensure the quality of the documents in KuatIA, we developed a processing pipeline inspired by previous dataset curation works. This pipeline is in active development, so different versions of KuatIA will have varying characteristics, including the number of documents and their contents. 

> 1. URL deduplication
> 2. MinHash deduplication

To find out more about our pipeline, we direct you to our repository

## Files and Subsets 🗃️

KuatIA is available to you as JSONL files, with each document (entry) consisting on a single JSON object. We provide these files with different structures to match different possible use cases.

### Curated Content 📇

The content that has been determined as "useful" by our pipeline is available through the `datasets` library. 

- Containing content from a single source dataset
- Compiling all documents from all datasets
- Containing all heuristic metrics, as specified in the **`Formatting and Data Fields 📑`** section, these subsets are addressed with the postfix `_full`
- Containing only an id and the text, these subsets are addressed with the postfix `_lite`

#### Usage

```python
import datasets
```

### Discarded Content 🗑️

Additionally, we provide you with the contents that have been discarded in each step of our pipeline. These are available in the repository but are not listed as subsets, and as such are not accessible through the datasets library. You can find any discarded content in the 

## Latest Changes

## Version History 🗄️



## Future Work

We plan to update KuatIA with the 

## Cite Us