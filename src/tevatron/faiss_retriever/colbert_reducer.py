import glob
import faiss
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Iterable, Tuple
from numpy import ndarray
import pickle
import numpy as np
import collections


def pickle_load(path):
    with open(path, 'rb') as f:
        scores, indices = pickle.load(f)
    return np.array(scores), indices.astype('int')


def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
    rh = None
    for scores, indices in results:
        if rh is None:
            print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
            rh = faiss.ResultHeap(scores.shape[0], scores.shape[1])
        rh.add_result(-scores, indices)
    rh.finalize()
    corpus_scores, corpus_indices = -rh.D, rh.I

    return corpus_scores, corpus_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query_id', required=True)
    parser.add_argument('--passage_id', required=True)
    parser.add_argument('--save_ranking_to', required=True)
    args = parser.parse_args()


    partitions = glob.glob(f'{args.score_dir}/*')

    corpus_scores, corpus_indices = combine_faiss_results(map(pickle_load, tqdm(partitions)))
    print(corpus_scores.shape, corpus_indices.shape)

    q_lookup = np.memmap(args.query_id, dtype=np.int32, mode="r")
    print(len(q_lookup))

    id_files = glob.glob(args.passage_id)
    id_files.sort()
    print(id_files)
    p_lookup = []
    for id_file in tqdm(id_files, total=len(id_files)):
        lookup = np.memmap(id_file, dtype=np.int32, mode="r")
        p_lookup.extend(list(lookup))
    print(len(p_lookup))


    q_result = collections.defaultdict(set)
    for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
        q_doc_indices = [p_lookup[i] for i in q_doc_indices]
        q_result[qid].update(q_doc_indices)
    
    for qid, docids in q_result.items():
        q_result[qid] = list(docids)
    print(len(q_result), np.mean([len(v) for v in q_result.values()]))

    with open(args.save_ranking_to, 'w') as outputfile:
        for qid, neighbors in q_result.items():
            for pid in neighbors:
                outputfile.write(f"{qid}\t{pid}\t0.0\n")   


if __name__ == '__main__':
    main()
