import json
import os
import pickle
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from datetime import datetime
from multiprocessing import Pool
from .utils import canonicalize_text
from ..tokenization import tokenize


SIMILARITY_THRESHOLD = 0.8
NUM_PERMS = 128


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
    
    Example:
        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> shingles = get_shingles(text, size=5)
        >>> # Returns shingles like: {"The quick brown fox jumps", 
        >>> #                        "quick brown fox jumps over", ...}
    """
    shingles = set()
    text_list = list(tokenize(text))
    if len(text_list) < size:
        return set([' '.join(text_list)])
    for i in range(len(text_list) - size + 1):
        shingle = ' '.join(text_list[i:i + size])
        shingles.add(shingle)
    return shingles


def read_corpora(corpora_dir):
    raw_corpora_path = os.path.join(corpora_dir, 'raw_corpora.jsonl')
    if not os.path.exists(raw_corpora_path):
        corpora = []
        corpus_files = [f for f in os.listdir(corpora_dir) if os.path.isfile(os.path.join(corpora_dir, f)) and f.endswith('.jsonl')]
        for corpus_file in tqdm(corpus_files, desc='Building raw corpora'):
            corpus_name = os.path.splitext(corpus_file)[0]
            corpus = []
            with open(os.path.join(corpora_dir, corpus_file), 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    record = json.loads(line)
                    text = record.get('text', '')
                    if text:
                        record['id'] = f'{corpus_name}_{i}'
                        record['duplicate'] = {'minhash': {'has_duplicate': False, 'dup_docs': []}}
                        corpus.append(record)
                    else:
                        print(f'Warning: Empty text in {corpus_name} at line {i}')
            corpora.extend(corpus)
            print(f'Finished processing {corpus_name}, which has {len(corpus)} documents')
        # save raw corpora to jsonl file
        with open(raw_corpora_path, 'w', encoding='utf-8') as f:
            for record in corpora:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print_path = '/'.join(raw_corpora_path.split('/')[-3:])
        print(f'Raw corpora with {len(corpora)} documents saved into {print_path}')
    else:
        corpora = []
        with open(raw_corpora_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                corpora.append(record)
    return corpora


def compute_minhash_row(row):
    text = row['text'] 
    can_text = canonicalize_text(text)
    shingles = get_shingles(can_text)
    text_minhash = MinHash(num_perm=NUM_PERMS)
    for shingle in shingles:
        text_minhash.update(shingle.encode('utf-8'))
    return {
        'id': row['id'], 
        'text': text, 
        'minhash': text_minhash, 
        'shingles': list(shingles)
    }


def compute_minhash(corpora, dedup_dir):
    u_corpora = []
    cpu_count = os.cpu_count()
    num_workers = max(1, (cpu_count - 1) if cpu_count else 1)
    with Pool(num_workers) as pool:
        u_corpora = list(tqdm(
            pool.imap_unordered(compute_minhash_row, [doc for doc in corpora]), 
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


def compute_lsh(corpora, dedup_dir):
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMS)
    for doc in tqdm(corpora, desc='Building LSH'):
        lsh.insert(doc['id'], doc['minhash'], check_duplication=False)
    lsh_path = os.path.join(dedup_dir, 'lsh.pkl')
    with open(lsh_path, 'wb') as f:
        pickle.dump(lsh, f)
    print_path = '/'.join(lsh_path.split('/')[-4:])
    print(f'LSH saved into {print_path}')
    return lsh


def find_duplicates(corpora, lsh):
    dup_dict = {}
    for doc in tqdm(corpora, desc='Finding duplicates'):
        id = doc['id']
        shingles = doc['shingles']
        text_minhash = MinHash(num_perm=NUM_PERMS)
        for shingle in shingles:
            text_minhash.update(shingle.encode('utf-8'))
        dups = lsh.query(text_minhash)
        dups = [dup for dup in dups if dup != id]
        dup_dict[id] = dups
    return dup_dict


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


def create_duplication_report(num_duplicates, corpora_len, dedup_dir, shingle_size):
    report = {
        'total_docs': corpora_len,
        'total_duplicates': num_duplicates,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'num_permutations': NUM_PERMS,
        'shingle_size': shingle_size,
    }
    # save report to json file
    report_path = os.path.join(dedup_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print_path = '/'.join(report_path.split('/')[-4:])
    print(f'Duplication report saved into {print_path}')


def update_corpora_metadata(corpora, duplicates, data_dir):
    corpora_lookup = {doc['id']: doc for doc in corpora}
    for dup in duplicates:
        id = dup['id']
        corpora_lookup[id]['duplicate']['minhash']['has_duplicate'] = True
        corpora_lookup[id]['duplicate']['minhash']['dup_docs'] = dup['dup_id']
    # save updated corpora to jsonl file
    updated_corpora_path = os.path.join(data_dir, 'processed', 'raw_corpora.jsonl')
    with open(updated_corpora_path, 'w', encoding='utf-8') as f:
        for doc in corpora:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print_path = '/'.join(updated_corpora_path.split('/')[-4:])
    print(f'Updated corpora with duplication metadata saved into {print_path}')


def check_duplicates(dup_dict, corpora, dedup_dir, shingle_size):
    # check jaccard similarity for each pair of duplicates
    corpora_lookup = {doc['id']: doc for doc in corpora}
    duplicates = []
    for id, dups in tqdm(dup_dict.items(), desc='Checking duplicates'):
        if dups:
            base_rec = corpora_lookup[id]
            base_text = base_rec['text']
            base_shingles = base_rec['shingles']
            for dup in dups:
                dup_rec = corpora_lookup[dup]
                dup_text = dup_rec['text']
                dup_shingles = dup_rec['shingles']
                similarity = jaccard_similarity(base_shingles, dup_shingles)
                if similarity >= SIMILARITY_THRESHOLD:
                    duplicates.append(
                        {
                            'id': id, 
                            'id_text': base_text, 
                            'dup_id': dup, 
                            'dup_text': dup_text,
                            'similarity': similarity
                        }
                    )
    print(f'Total duplicates found (similarity threshold {SIMILARITY_THRESHOLD}): {len(duplicates)}')
    # save duplicates to json file
    dup_path = os.path.join(dedup_dir, 'duplicates.json')
    with open(dup_path, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, indent=4, ensure_ascii=False)
    print_path = '/'.join(dup_path.split('/')[-4:])
    print(f'Duplicates saved into {print_path}')
    create_duplication_report(len(duplicates), len(corpora), dedup_dir, shingle_size)
    return duplicates


def run_deduplication(data_dir, shingle_size=5):
    # create deduplication directory if it doesn't exist
    dt = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(os.path.join(data_dir, 'deduplication', dt), exist_ok=True)
    dedup_dir = os.path.join(data_dir, 'deduplication', dt)
    # read corpora
    corpora_dir = os.path.join(data_dir, 'processed')
    corpora = read_corpora(corpora_dir)
    # compute minhash
    # u_corpora = compute_minhash(corpora, dedup_dir)
    # compute lsh
    # lsh = compute_lsh(u_corpora, dedup_dir)
    # find duplicates
    # duplicates = find_duplicates(u_corpora, lsh)
    # check duplicates
    # duplicates = check_duplicates(duplicates, u_corpora, dedup_dir, shingle_size)
    # update corpora metadata with duplication info
    # update_corpora_metadata(corpora, duplicates, data_dir)
    print('Deduplication has successfully completed')


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, 'data')
    run_deduplication(data_dir)