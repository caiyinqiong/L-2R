import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import faiss
import torch
import json
import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--buffer_file', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--old_buffer_emb_file', default=None)
    parser.add_argument('--new_buffer_emb_file', required=True)

    args = parser.parse_args()

    old_buffer_emb = {}
    if args.old_buffer_emb_file:
        with open(args.old_buffer_emb_file, 'rb') as f:
            old_buffer_emb = pickle.load(f)

    buffer_dids = set()
    with open(args.buffer_file, 'rb') as f:
        _, buffer_qid2dids = pickle.load(f)
    for qid, dids in buffer_qid2dids.items():
        buffer_dids.update(dids)
    with open(args.train_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            docs = data['positive_passages']
            buffer_dids.update([doc['docid'] for doc in docs])
    print('total buffer doc: ', len(buffer_dids))

    buffer_emb = {}
    for did in buffer_dids:
        if did in old_buffer_emb:
            buffer_emb[did] = old_buffer_emb[did]

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')
    for file in tqdm(index_files, total=len(index_files)):
        with open(file, 'rb') as f:
            p_reps, p_lookup = pickle.load(f)
            for (rep, look_up) in zip(p_reps, p_lookup):
                if look_up in buffer_dids:
                    buffer_emb[look_up] = rep

    output = open(os.path.join(args.new_buffer_emb_file, 'buffer_emb.pkl'), 'wb')
    pickle.dump(buffer_emb, output)
    output.close()
