# 支持：从(threshold_rank, top]中采样sample个样本
import json
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Manager
from tqdm import tqdm
import numpy as np
from pyserini.eval.evaluate_dpr_retrieval import SimpleTokenizer, has_answers

np.random.seed(42)

query2qrel = {}
with open('/data/users/caiyingqiong/lifelong_learning/lifelong_lotte/data_incre_aug/train_dir/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        query2qrel[data['query_id']] = set([pos['docid'] for pos in data['positive_passages']])


class BasicHardNegativeMiner:
    def __init__(self, results_path, corpus_dataset, sample, threshold_rank, top):
        self.corpus_data = corpus_dataset
        self.sample=sample
        self.threshold_rank = threshold_rank
        self.top = top
        self.retrieval_results = self._read_result(results_path, threshold_rank, top)
        self.docid_to_idx = {k: v for v, k in enumerate(self.corpus_data['docid'])}
        print(len(self.retrieval_results), np.mean([len(res) for res in self.retrieval_results.values()]))

    @staticmethod
    def _read_result(path, threshold_rank, top):
        retrieval_results = {}
        with open(path) as f:
            for line in tqdm(f):
                qid, _, pid, rank, _, _ = line.rstrip().split()
                qid, pid = int(qid), int(pid)
                if int(rank) > threshold_rank and int(rank) <= top:
                    if qid not in retrieval_results:
                        retrieval_results[qid] = []
                    retrieval_results[qid].append(pid)
        return retrieval_results

    def __call__(self, example):
        query_id = example['query_id']
        if query_id in self.retrieval_results:
            retrieved_docid = self.retrieval_results[query_id]
            # positive_ids = [pos['docid'] for pos in example['positive_passages']]
            hard_negatives = []
            for docid in np.random.choice(retrieved_docid, min(self.sample, len(retrieved_docid)), replace=False):
                doc_info = self.corpus_data[self.docid_to_idx[docid]]
                text = doc_info['text']
                title = doc_info['title'] if 'title' in doc_info else None
                if docid not in query2qrel[query_id]:    # positive_ids:
                    hn_doc = {'docid': docid, 'text': text}
                    if title:
                        hn_doc['title'] = title
                    hard_negatives.append(hn_doc)
            example['negative_passages'] = hard_negatives
            return example
        else:
            print(query_id)
            example['negative_passages'] = []
            return example


class EMHardNegativeMiner(BasicHardNegativeMiner):
    def __init__(self, results_path, corpus_dataset, depth, tokenzier, regex=False):
        self.tokenizer = tokenzier
        self.regex = regex
        super().__init__(results_path, corpus_dataset, depth)

    def __call__(self, example):
        query_id = example['query_id']
        retrieved_docid = self.retrieval_results[query_id]
        answers = example['answers']
        positives = []
        hard_negatives = []
        for docid in retrieved_docid[:self.depth]:
            doc_info = self.corpus_data[self.docid_to_idx[docid]]
            text = doc_info['text']
            title = doc_info['title'] if 'title' in doc_info else None
            if not has_answers(text, answers, self.tokenizer, self.regex):
                hn_doc = {'docid': docid, 'text': text}
                if title:
                    hn_doc['title'] = title
                hard_negatives.append(hn_doc)
            else:
                pos_doc = {'docid': docid, 'text': text}
                if title:
                    pos_doc['title'] = title
                positives.append(pos_doc)
        example['negative_passages'] = hard_negatives
        example['positive_passages'] = positives
        return example


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data_name', type=str, required=True)
    parser.add_argument('--corpus_data_name', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--top', type=int, default=1000, required=True)
    parser.add_argument('--sample', type=int, default=100, required=True)
    parser.add_argument('--threshold_rank', type=int, default=0, required=False)
    parser.add_argument('--min_hn', type=int, default=10, required=False)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=False)
    parser.add_argument('--proc_num', type=int, default=12, required=False)
    parser.add_argument('--em', action='store_true', required=False)
    parser.add_argument('--regex', action='store_true', required=False)

    args = parser.parse_args()
    train_data = load_dataset('json', data_files={'train':args.train_data_name}, cache_dir=args.cache_dir)['train']
    corpus_data = load_dataset('json', data_files={'train':args.corpus_data_name}, cache_dir=args.cache_dir)['train']
    if args.em:
        miner = EMHardNegativeMiner(args.result_path, corpus_data, args.sample, SimpleTokenizer(), regex=args.regex)
    else:
        miner = BasicHardNegativeMiner(args.result_path, corpus_data, args.sample, args.threshold_rank, args.top)

    hn_data = train_data.map(
        miner,
        batched=False,
        # num_proc=args.proc_num,
        desc="Running hard negative mining",
    )

    combined_data = hn_data  # concatenate_datasets([train_data, hn_data])
    combined_data = combined_data.filter(
        function=lambda data: len(data["positive_passages"]) >= 1 and len(data["negative_passages"]) >= args.min_hn
    )

    with open(args.output, 'w') as f:
        for e in tqdm(combined_data):
            f.write(json.dumps(e, ensure_ascii=False)+'\n')
