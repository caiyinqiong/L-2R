import numpy as np
import glob
from argparse import ArgumentParser
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_id', required=True)
    parser.add_argument('--passage_id', required=True)
    parser.add_argument('--save_idmap_to', required=True)

    args = parser.parse_args()

    id_files = glob.glob(args.passage_id)
    id_files.sort()
    print(id_files)
    logger.info(f'Pattern match found {len(id_files)} files; loading them into index.')

    shards = []
    for id_file in tqdm(id_files, total=len(id_files)):
        lookup = np.memmap(id_file, dtype=np.int32, mode="r")
        shards.append(lookup)
    
    lenn = [len(lookup) for lookup in shards]
    print(np.sum(lenn))

    import os
    start = 0
    for i in range(len(shards)):
        idmap_memmap = np.memmap(os.path.join(args.save_idmap_to, f'corpus_emb.{str(i).zfill(2)}.idmap'), 
            dtype=np.int32, mode="w+", shape=(lenn[i], ))
        data = np.array(range(start, start+ lenn[i]))
        idmap_memmap[:] = data
        start += lenn[i]
    
    q_lookup = np.memmap(args.query_id, dtype=np.int32, mode="r")
    idmap_memmap = np.memmap(os.path.join(args.save_idmap_to, f'query_emb.idmap'), 
            dtype=np.int32, mode="w+", shape=(len(q_lookup), ))
    print(len(q_lookup))
    data = np.array(range(len(q_lookup)))
    idmap_memmap[:] = data



if __name__ == '__main__':
    main()
