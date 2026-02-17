import json
import os
import pandas as pd
import pickle
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from datetime import datetime
from multiprocessing import Pool
from .utils import canonicalize_text
from ..tokenization import tokenize


SIMILARITY_THRESHOLD = 0.8
NUM_PERMS = 128
SHINGLE_SIZE = 5


def get_shingles(text):
    shingles = set()
    text_list = list(tokenize(text))
    if len(text_list) < SHINGLE_SIZE:
        return set([' '.join(text_list)])
    for i in range(len(text_list) - SHINGLE_SIZE + 1):
        shingle = ' '.join(text_list[i:i + SHINGLE_SIZE])
        shingles.add(shingle)
    return shingles


def read_corpora(corpora_dir, dedup_dir):
    corpora = []
    corpus_files = [f for f in os.listdir(corpora_dir) if os.path.isfile(os.path.join(corpora_dir, f))]
    for corpus_file in tqdm(corpus_files, desc='Building corpora'):
        corpus_name = os.path.splitext(corpus_file)[0]
        with open(os.path.join(corpora_dir, corpus_file), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                text = record.get('text', '')
                if text:
                    corpora.append({'id': f'{corpus_name}_{i}', 'text': text})
    # save raw corpora to jsonl file
    raw_corpora_path = os.path.join(dedup_dir, 'raw_corpora.jsonl')
    with open(raw_corpora_path, 'w', encoding='utf-8') as f:
        for record in corpora:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print_path = '/'.join(raw_corpora_path.split('/')[-4:])
    print(f'Raw corpora saved into {print_path}')
    corpora_df = pd.DataFrame(corpora)
    return corpora_df


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


def compute_minhash(corpora_df, dedup_dir):
    u_corpora = []
    cpu_count = os.cpu_count()
    num_workers = max(1, (cpu_count - 1) if cpu_count else 1)
    with Pool(num_workers) as pool:
        u_corpora = list(tqdm(
            pool.imap_unordered(compute_minhash_row, [row for _, row in corpora_df.iterrows()]), 
            total=len(corpora_df), 
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


def check_duplicates(dup_dict, corpora, dedup_dir):
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
    # save duplicates to json file
    dup_path = os.path.join(dedup_dir, 'duplicates.json')
    with open(dup_path, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, indent=4, ensure_ascii=False)
    print_path = '/'.join(dup_path.split('/')[-4:])
    print(f'Duplicates saved into {print_path}')
    return duplicates


def run_deduplication(data_dir):
    # create deduplication directory if it doesn't exist
    dt = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(os.path.join(data_dir, 'deduplication', dt), exist_ok=True)
    dedup_dir = os.path.join(data_dir, 'deduplication', dt)
    # read corpora
    corpora_dir = os.path.join(data_dir, 'processed')
    corpora = read_corpora(corpora_dir, dedup_dir)
    # compute minhash
    corpora = compute_minhash(corpora, dedup_dir)
    # compute lsh
    lsh = compute_lsh(corpora, dedup_dir)
    # find duplicates
    duplicates = find_duplicates(corpora, lsh)
    # check duplicates
    duplicates_df = check_duplicates(duplicates, corpora, dedup_dir)
    return duplicates_df


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, 'data')
    duplicates_df = run_deduplication(data_dir)