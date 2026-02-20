import json
import os
import pickle
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from .utils import canonicalize_text
from ..tokenization import tokenize



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


def read_corpora(corpora_dir: str, dedup_dir: str, update_corpora: bool = False) -> list[dict]:
    """
    Read and aggregate corpus files from a directory, with caching support.
    
    This function reads JSONL files from the specified directory and aggregates them
    into a single corpora list. If a 'raw_corpora.jsonl' cache file already exists,
    it reads from the cache instead of re-processing all files. Each record is
    augmented with a unique ID and duplicate tracking metadata.
    
    Args:
        corpora_dir (str): Path to the directory containing corpus JSONL files.
                          The function will look for .jsonl files and optionally
                          use a cached 'raw_corpora.jsonl' file.
    
    Returns:
        list[dict]: A list of document dictionaries, where each dictionary contains:
                    - 'text' (str): The document text content.
                    - 'id' (str): Unique identifier in format 'corpus_name_line_number'.
                    - 'duplicate' (dict): Metadata tracking duplicate status and related docs.
                    - Other fields from the original JSONL record.
    
    Notes:
        - Empty text records are skipped with a warning message.
        - The function creates a 'raw_corpora.jsonl' cache file for faster
          subsequent reads.
        - Progress bar shown during corpus reading via tqdm.
    """
    raw_corpora_path = os.path.join(dedup_dir, 'raw_corpora.jsonl')
    if not os.path.exists(raw_corpora_path) or update_corpora:
        corpora = []
        corpus_files = [f for f in os.listdir(corpora_dir) if os.path.isfile(os.path.join(corpora_dir, f)) and f.endswith('.jsonl')]
        for corpus_file in tqdm(corpus_files, desc='Building raw corpora'):
            corpus_name = os.path.splitext(corpus_file)[0]
            corpus = []
            corpus_words = 0
            corpus_chars = 0
            with open(os.path.join(corpora_dir, corpus_file), 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    record = json.loads(line)
                    text = record.get('text', '')
                    if text:
                        if 'duplicate' not in record:
                            record['duplicate'] = {'minhash': {'has_duplicate': False}}
                        else:
                            record['duplicate'].update({'minhash': {'has_duplicate': False}})
                        corpus.append(record)
                        corpus_words += record['num_words_split']
                        corpus_chars += record['num_chars']
                    else:
                        print(f'Warning: Empty text in {corpus_name} at line {i}')
            #print(f'Processing corpus: {corpus_name} with {len(corpus)} documents, '\
            #      f'{corpus_words} words, {corpus_chars} characters')
            corpora.extend(corpus)
        # save raw corpora to jsonl file
        with open(raw_corpora_path, 'w', encoding='utf-8') as f:
            for record in corpora:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print_path = '/'.join(raw_corpora_path.split('/')[-3:])
        print(f'Raw corpora with {len(corpora)} documents saved into {print_path}')
    else:
        print(f'Loading raw corpora from cache at {raw_corpora_path}')
        corpora = []
        with open(raw_corpora_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                record['duplicate'] = {'minhash': {'has_duplicate': False}}
                corpora.append(record)
    return corpora


def compute_minhash_row(row: dict, num_perm: int) -> dict:
    """
    Compute MinHash signature and shingles for a single document.
    
    This function processes a document record by canonicalizing its text,
    extracting shingles (n-grams), and computing a MinHash signature.
    The MinHash signature is used for efficient duplicate detection via LSH.
    
    Args:
        row (dict): A document dictionary with at least the following keys:
                    - 'id' (str): Unique document identifier.
                    - 'text' (str): Document text content.
    
    Returns:
        dict: A dictionary containing:
              - 'id' (str): Document identifier (from input).
              - 'text' (str): Original document text.
              - 'minhash' (datasketch.MinHash): Computed MinHash signature with NUM_PERMS hash functions.
              - 'shingles' (list[str]): List of shingles (space-separated token n-grams).
    
    Notes:
        - Text is first canonicalized (normalized, lowercased, punctuation removed).
        - Shingles are extracted from canonicalized text with size SHINGLE_SIZE.
        - MinHash is computed using all shingles; each shingle is encoded as UTF-8
          before hashing.
        - Used in parallel processing for efficient batch computation.
    """
    text = row['text'] 
    can_text = canonicalize_text(text)
    shingles = get_shingles(can_text)
    text_minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        text_minhash.update(shingle.encode('utf-8'))
    return {
        'id': row['id'], 
        'text': text, 
        'minhash': text_minhash, 
        'shingles': list(shingles)
    }


def compute_minhash(corpora: list[dict], dedup_dir: str, num_perm: int) -> list[dict]:
    """
    Compute MinHash signatures for all documents using parallel processing.
    
    This function processes a list of documents in parallel to compute MinHash signatures
    and extract shingles for each document. Results are saved to a JSON file and returned.
    Parallelization uses multiprocessing.Pool for efficient CPU utilization.
    
    Args:
        corpora (list[dict]): List of document dictionaries, each with at least:
                              - 'id' (str): Unique document identifier.
                              - 'text' (str): Document text content.
        dedup_dir (str): Directory path where the 'minhash.json' output file will be saved.
    
    Returns:
        list[dict]: List of processed document dictionaries, each containing:
                    - 'id' (str): Document identifier.
                    - 'text' (str): Original document text.
                    - 'minhash' (datasketch.MinHash): Computed MinHash signature.
                    - 'shingles' (list[str]): Extracted shingles from the document.
    
    Notes:
        - Uses multiprocessing.Pool with (cpu_count - 1) workers for parallelization.
        - Falls back to 1 worker if cpu_count() returns None.
        - Progress tracked via tqdm bar during processing.
        - MinHash signatures are excluded from JSON output (not JSON-serializable).
        - Output JSON file contains one record per line (JSONL format).
    """
    u_corpora = []
    cpu_count = os.cpu_count()
    num_workers = max(1, (cpu_count - 1) if cpu_count else 1)
    with Pool(num_workers) as pool:
        worker = partial(compute_minhash_row, num_perm=num_perm)
        u_corpora = list(tqdm(
            pool.imap_unordered(worker, corpora), 
            total=len(corpora), 
            desc='Computing MinHash'
        ))
    # save minhash to json file
    minhash_path = os.path.join(dedup_dir, 'minhash.json')
    with open(minhash_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps({k: v for k, v in r.items() if k != 'minhash'}, default=str, ensure_ascii=False) for r in u_corpora))
    print_path = '/'.join(minhash_path.split('/')[-4:])
    print(f'Minhash saved into {print_path}')
    return u_corpora


def compute_lsh(corpora: list[dict], dedup_dir: str, similarity_threshold: float=0.8, num_perm: int=128) -> MinHashLSH:
    """
    Build a Locality-Sensitive Hashing (LSH) index from document MinHash signatures.
    
    This function constructs an LSH index by inserting MinHash signatures from all
    documents. LSH allows efficient approximate nearest-neighbor queries to find
    similar documents without comparing every pair. The index is saved to disk as
    a pickle file.
    
    Args:
        corpora (list[dict]): List of document dictionaries, each containing:
                              - 'id' (str): Unique document identifier.
                              - 'minhash' (datasketch.MinHash): Precomputed MinHash signature.
        dedup_dir (str): Directory path where the 'lsh.pkl' index file will be saved.
        similarity_threshold (float): Jaccard similarity threshold for LSH (default 0.8).
        num_perm (int): Number of permutations (hash functions) used in MinHash (default
    
    Returns:
        MinHashLSH: The constructed LSH index that can be queried to find candidate
                    duplicate documents. The index uses similarity_threshold and num_perm
                    as configuration parameters.
    
    Notes:
        - Uses similarity_threshold and num_perm parameters for LSH configuration.
        - Progress tracked via tqdm bar during index construction.
        - check_duplication is set to False for efficiency (no deduplication during insert).
        - The index is persisted to disk via pickle serialization.
    """
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    for doc in tqdm(corpora, desc='Building LSH'):
        lsh.insert(doc['id'], doc['minhash'], check_duplication=False)
    lsh_path = os.path.join(dedup_dir, 'lsh.pkl')
    with open(lsh_path, 'wb') as f:
        pickle.dump(lsh, f)
    print_path = '/'.join(lsh_path.split('/')[-4:])
    print(f'LSH saved into {print_path}')
    return lsh


def find_duplicates(corpora: list[dict], lsh: MinHashLSH, num_perm: int) -> dict[str, list[str]]:
    """
    Find candidate duplicate documents using LSH index queries.
    
    This function queries the LSH index for each document to retrieve candidate
    duplicates. Each document's MinHash signature is recomputed from its shingles
    and used to query the index for similar documents.
    
    Args:
        corpora (list[dict]): List of document dictionaries, each containing at least:
                              - 'id' (str): Unique document identifier.
                              - 'shingles' (list[str]): Precomputed list of shingles.
        lsh (MinHashLSH): Precomputed LSH index for similarity queries.
        num_perm (int): Number of permutations (hash functions) used in MinHash.
                        Must match the value used in lsh construction.
    
    Returns:
        dict[str, list[str]]: Dictionary mapping document IDs to lists of candidate
                              duplicate document IDs. Self-duplicates (id matching itself)
                              are excluded from results.
    
    Notes:
        - Progress tracked via tqdm bar during query execution.
        - MinHash signatures are recomputed for each document during query.
        - LSH queries return approximate nearest neighbors (candidates), not exact duplicates.
        - Exact duplicate verification requires additional similarity computation (Jaccard).
    """
    dup_dict = {}
    for doc in tqdm(corpora, desc='Finding duplicates'):
        id = doc['id']
        shingles = doc['shingles']
        text_minhash = MinHash(num_perm=num_perm)
        for shingle in shingles:
            text_minhash.update(shingle.encode('utf-8'))
        dups = lsh.query(text_minhash)
        dups = [dup for dup in dups if dup != id]
        dup_dict[id] = dups
    return dup_dict


def jaccard_similarity(list1: list[str], list2: list[str]) -> float:
    """
    Compute the Jaccard similarity coefficient between two lists.
    
    Jaccard similarity measures the overlap between two sets as the ratio of
    intersection size to union size. It ranges from 0 (no overlap) to 1 (identical sets).
    Used for exact duplicate verification after LSH candidate retrieval.
    
    Args:
        list1 (list[str]): First list of elements (typically shingles).
        list2 (list[str]): Second list of elements (typically shingles).
    
    Returns:
        float: Jaccard similarity coefficient in range [0, 1].
               Returns 0 if both lists are empty.
    
    Notes:
        - Converts input lists to sets, so duplicates within lists are ignored.
        - Formula: |A ∩ B| / |A ∪ B|
        - Used to verify exact duplicate matches after LSH approximate search.
    """
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


def create_duplication_report(num_duplicates: int, corpora_len: int, dedup_dir: str, 
                              shingle_size: int, num_perm: int, 
                              similarity_threshold: float, start_time: str, 
                              end_time: str) -> None:
    """
    Generate and save a deduplication report with statistics and configuration.
    
    Creates a summary report of the deduplication process, including counts of
    documents and duplicates found, as well as the algorithm parameters used.
    The report is saved as a JSON file in the deduplication directory.
    
    Args:
        num_duplicates (int): Total number of duplicate pairs found.
        corpora_len (int): Total number of documents in the corpus.
        dedup_dir (str): Directory path where the 'report.json' file will be saved.
        shingle_size (int): Size of shingles used (number of tokens per shingle).
        num_perm (int): Number of permutations (hash functions) used in MinHash.
        similarity_threshold (float): Similarity threshold used for LSH queries
                                      (range typically 0.0 to 1.0).
    
    Returns:
        None. The function creates a report dictionary and saves it to disk as 'report.json'.
    
    Notes:
        - Report is persisted to disk as 'report.json' in JSONL-like format.
        - Unicode characters are preserved (ensure_ascii=False).
        - Provides reproducibility information for the deduplication run.
    """
    report = {
        'start_time': start_time,
        'end_time': end_time,
        'duration_minutes': (datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')).total_seconds() / 60,
        'total_docs': corpora_len,
        'total_duplicates': num_duplicates,
        'similarity_threshold': similarity_threshold,
        'num_permutations': num_perm,
        'shingle_size': shingle_size,
    }
    # save report to json file
    report_path = os.path.join(dedup_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    print_path = '/'.join(report_path.split('/')[-4:])
    print(f'Duplication report saved into {print_path}')


def update_corpora_metadata(corpora: list[dict], duplicates: list[dict], 
                            data_dir: str, dedup_dir: str, corpora_dir: str) -> None:
    """
    Update corpus documents with duplicate metadata and persist to disk.
    
    Augments each document in the corpora with information about its duplicates.
    For each duplicate pair found, marks the source document as having duplicates
    and lists the duplicate document IDs. Updated corpora are saved back to the
    processed directory.
    
    Args:
        corpora (list[dict]): List of document dictionaries, each containing:
                              - 'id' (str): Unique document identifier.
                              - 'duplicate' (dict): Metadata structure with nested keys
                                'minhash'['has_duplicate'] and 'minhash'['dup_docs'].
                              - Other document fields.
        duplicates (list[dict]): List of duplicate pair dictionaries, each containing:
                                 - 'id' (str): Source document ID.
                                 - 'dup_id' (str): Duplicate document ID.
        data_dir (str): Base data directory path. Updated corpora are saved to
                        data_dir/processed/raw_corpora.jsonl.
    
    Returns:
        None. Modifies corpora in-place and persists changes to disk.
    
    Notes:
        - Creates a lookup dictionary for O(1) document access.
        - Sets duplicate["minhash"]["has_duplicate"] = True for documents with duplicates.
        - Stores duplicate IDs in duplicate["minhash"]["dup_docs"] field.
        - Overwrites the raw_corpora.jsonl file with updated metadata.
        - Unicode characters are preserved in output (ensure_ascii=False).
    """
    corpora_lookup = {doc['id']: doc for doc in corpora}
    duplicate_docs = {}
    for dup in duplicates:
        id = dup['id']
        corpora_lookup[id]['duplicate']['minhash']['has_duplicate'] = True
        if id not in duplicate_docs:
            duplicate_docs[id] = set()
        duplicate_docs[id].add(dup['dup_id'])
    
    # save updated corpora to jsonl file
    updated_corpora_path = os.path.join(dedup_dir, 'raw_corpora.jsonl')
    with open(updated_corpora_path, 'w', encoding='utf-8') as f:
        for doc in corpora:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print_path = '/'.join(updated_corpora_path.split('/')[-4:])
    print(f'Updated corpora with duplication metadata saved into {print_path}')
    
    # update each corpus file with duplication metadata
    corpus_files = [f for f in os.listdir(corpora_dir) \
        if os.path.isfile(os.path.join(corpora_dir, f)) and f.endswith('.jsonl')]
    for corpus_file in tqdm(corpus_files, desc='Updating corpora with duplication metadata'):
        corpus_name = os.path.splitext(corpus_file)[0]
        updated_corpus_path = os.path.join(corpora_dir, corpus_file)
        with open(updated_corpus_path, 'w', encoding='utf-8') as f:
            docs = [doc for doc in corpora if doc['id'].startswith(f'{corpus_name}_')]
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print_path = '/'.join(updated_corpus_path.split('/')[-4:])
    
    # save duplicate docs to json file
    dup_docs_path = os.path.join(dedup_dir, 'duplicates.json')
    with open(dup_docs_path, 'w', encoding='utf-8') as f:
        # convert sets to lists for JSON serialization
        for id, dup_ids in duplicate_docs.items():
            duplicate_docs[id] = list(dup_ids)
        json.dump(duplicate_docs, f, indent=4, ensure_ascii=False)
    print_path = '/'.join(dup_docs_path.split('/')[-4:])
    print(f'Duplicate docs saved into {print_path}')


def save_duplicates(duplicates: list[dict], dedup_dir: str, batch_size: int = 1000000) -> None:
    """
    Save confirmed duplicate pairs to a JSON file in the deduplication directory.
    
    This function takes a list of verified duplicate pairs and saves them to a
    'duplicates.json' file in the specified deduplication directory. Each pair
    includes document IDs, texts, and similarity scores. The output is formatted
    for readability and preserves Unicode characters.
    
    Args:
        duplicates (list[dict]): List of duplicate pair dictionaries, each containing:
                                 - 'id' (str): Source document ID.
                                 - 'id_text' (str): Source document text.
                                 - 'dup_id' (str): Duplicate document ID.
                                 - 'dup_text' (str): Duplicate document text.
                                 - 'similarity' (float): Jaccard similarity score.
        dedup_dir (str): Directory path where the 'duplicates.json' file will be saved.
        batch_size (int): Maximum number of duplicate pairs to save in a single JSON file.
                          If the number of duplicates exceeds this threshold, multiple files
                          will be created with suffixes (e.g., duplicates_0.json, duplicates_1.json).
    
    Returns:
        None. Modifies corpora in-place and persists changes to disk.
    """
    dup_dir = os.path.join(dedup_dir, 'duplicate_details')
    os.makedirs(dup_dir, exist_ok=True)
    num_duplicates = len(duplicates)
    if num_duplicates > batch_size:
        num_files = num_duplicates // batch_size + (1 if num_duplicates % batch_size else 0)
        for i in range(num_files):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_duplicates)
            batch = duplicates[start_idx:end_idx]
            dup_path = os.path.join(dup_dir, f'duplicates_{i}.json')
            with open(dup_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, indent=4, ensure_ascii=False)
            print_path = '/'.join(dup_path.split('/')[-4:])
            print(f'Duplicates saved into {print_path}')
    else:
        dup_path = os.path.join(dup_dir, 'duplicates.json')
        with open(dup_path, 'w', encoding='utf-8') as f:
            json.dump(duplicates, f, indent=4, ensure_ascii=False)
        print_path = '/'.join(dup_path.split('/')[-4:])
        print(f'Duplicates saved into {print_path}')


def check_duplicates(dup_dict: dict[str, list[str]], corpora: list[dict], 
                     dedup_dir: str, shingle_size: int, 
                     similarity_threshold: float, num_perm: int, 
                     start_time: str) -> list[dict]:
    """
    Verify duplicate candidates using Jaccard similarity and save confirmed duplicates.
    
    Filters LSH approximate duplicates by computing exact Jaccard similarity between
    candidate pairs. Only pairs exceeding the similarity threshold are marked as true
    duplicates. Results are saved to JSON file and a summary report is generated.
    
    Args:
        dup_dict (dict[str, list[str]]): Mapping of document IDs to lists of candidate
                                         duplicate document IDs from LSH queries.
        corpora (list[dict]): List of document dictionaries, each containing:
                              - 'id' (str): Unique document identifier.
                              - 'text' (str): Document text content.
                              - 'shingles' (list[str]): Precomputed shingles.
        dedup_dir (str): Directory path where duplicate results and report will be saved.
        shingle_size (int): Size of shingles used (for report metadata).
        similarity_threshold (float): Minimum Jaccard similarity (0.0-1.0) to classify
                                      a pair as true duplicates.
        num_perm (int): Number of MinHash permutations used (for report metadata).
    
    Returns:
        list[dict]: List of verified duplicate pairs, each containing:
                    - 'id' (str): Base document ID.
                    - 'id_text' (str): Base document text.
                    - 'dup_id' (str): Duplicate document ID.
                    - 'dup_text' (str): Duplicate document text.
                    - 'similarity' (float): Computed Jaccard similarity.
    
    Notes:
        - Exact similarity computation via Jaccard on shingle sets.
        - Saves duplicates.json and calls create_duplication_report() for summary.
        - Unicode characters preserved in JSON output (ensure_ascii=False).
    """
    # check jaccard similarity for each pair of duplicates
    corpora_lookup = {doc['id']: doc for doc in corpora}
    duplicates = []
    duplicate_docs = set()
    for id, dups in tqdm(dup_dict.items(), desc='Checking duplicates'):
        if dups:
            duplicate_docs.add(id)
            base_rec = corpora_lookup[id]
            base_text = base_rec['text']
            base_shingles = base_rec['shingles']
            for dup in dups:
                duplicate_docs.add(dup)
                dup_rec = corpora_lookup[dup]
                dup_text = dup_rec['text']
                dup_shingles = dup_rec['shingles']
                similarity = jaccard_similarity(base_shingles, dup_shingles)
                if similarity >= similarity_threshold:
                    duplicates.append(
                        {
                            'id': id, 
                            'id_text': base_text, 
                            'dup_id': dup, 
                            'dup_text': dup_text,
                            'similarity': similarity
                        }
                    )
    print(f'Total duplicates found (similarity threshold {similarity_threshold}): \
        {len(duplicate_docs)}')
    # save duplicates to json file
    save_duplicates(duplicates, dedup_dir)
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    create_duplication_report(len(duplicate_docs), len(corpora), dedup_dir, 
                              shingle_size, num_perm, similarity_threshold, 
                              start_time, end_time)
    return duplicates


def execute_minhash_deduplication(data_dir: str, output_dir: str, 
                                  shingle_size: int=5, 
                                  similarity_threshold: float=0.8, 
                                  num_perm: int=128):
    
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # create deduplication directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'deduplication'), exist_ok=True)
    dt = datetime.now().strftime('%Y%m%d_%H%M')
    dedup_dir = os.path.join(output_dir, 'deduplication', dt)
    os.makedirs(dedup_dir, exist_ok=True)
    # read corpora
    corpora_dir = os.path.join(data_dir, 'processed')
    corpora = read_corpora(corpora_dir, dedup_dir, update_corpora=False)
    # compute minhash
    u_corpora = compute_minhash(corpora, dedup_dir, num_perm)
    # compute lsh
    lsh = compute_lsh(u_corpora, dedup_dir, similarity_threshold, num_perm)
    # find duplicates
    duplicates = find_duplicates(u_corpora, lsh, num_perm)
    # check duplicates
    duplicates = check_duplicates(duplicates, u_corpora, dedup_dir, shingle_size, 
                                  similarity_threshold, num_perm, start_time)
    # update corpora metadata with duplication info
    update_corpora_metadata(corpora, duplicates, data_dir, dedup_dir, corpora_dir)
    print('Deduplication has successfully completed')
