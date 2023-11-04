import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import faiss
import torch

from .retriever import BaseFaissIPRetriever

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def mammap_load(index_path, id_path, dim):
    lookup = np.memmap(id_path, dtype=np.int32, mode="r")
    total = lookup.shape[0]
    reps = np.memmap(index_path, dtype=np.float32, mode="r")
    reps = reps.reshape(total, dim).astype('float32')
    return reps, list(lookup)


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--query_idmap', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--passage_idmap', required=True)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)

    args = parser.parse_args()

    logger.info('start load ....')
    p_reps, p_lookup = mammap_load(args.passage_reps, args.passage_idmap, args.dim)
    logger.info('start build ....')
    retriever = BaseFaissIPRetriever(p_reps)
    faiss.omp_set_num_threads(10)
    retriever.add(p_reps)
    look_up = p_lookup

    logger.info('convert_index_to_gpu ...')
    if torch.cuda.is_available():
        retriever.convert_index_to_gpu(list(range(torch.cuda.device_count())), useFloat16=True)
    else:
        faiss.omp_set_num_threads(32)

    logger.info('Index Search Start')
    q_reps, q_lookup = mammap_load(args.query_reps, args.query_idmap, args.dim)
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')
    print(all_scores.shape, psg_indices.shape)

    pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
