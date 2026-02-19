import os

from .minhash import execute_minhash_deduplication


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, 'data')
    execute_minhash_deduplication(data_dir)
    
    