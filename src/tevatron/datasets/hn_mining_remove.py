# 支持： 1）从[1, threshold_rank]中score<threshold_score的集合中采样sample_top个；2）从(threshold_rank, top]中采样出sample个样本
import json
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Manager
from tqdm import tqdm
import numpy as np
from pyserini.eval.evaluate_dpr_retrieval import SimpleTokenizer, has_answers

np.random.seed(42)

query2qrel = {}
with open('/data/sdd/caiyinqiong/lifelong_learning/tevatron/data/msmarco-passage/train_dir/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        query2qrel[data['query_id']] = set([pos['docid'] for pos in data['positive_passages']])


class BasicHardNegativeMiner:
    def __init__(self, results_path, rank_result_path, corpus_dataset, sample, sample_top, top, threshold_rank, threshold_score):
        self.corpus_data = corpus_dataset
        self.sample=sample
        self.sample_top=sample_top

        rank_results = self._read_rank_result(rank_result_path)
        self.retrieval_results, self.top_results = self._read_result(rank_results, results_path, threshold_rank, top, threshold_score)
        self.docid_to_idx = {k: v for v, k in enumerate(self.corpus_data['docid'])}
        print(len(self.retrieval_results), np.mean([len(res) for res in self.retrieval_results.values()]))
        print(len(self.top_results), np.mean([len(res) for res in self.top_results.values()]))

    @staticmethod
    def _read_result(rank_results, results_path, threshold_rank, top, threshold_score):
        retrieval_results = {}
        top_results = {}
        with open(results_path) as f:
            for line in tqdm(f):
                qid, _, pid, rank, _, _ = line.rstrip().split()
                if int(rank) > threshold_rank and int(rank) <= top:
                    if qid not in retrieval_results:
                        retrieval_results[str(qid)] = []
                    retrieval_results[str(qid)].append(str(pid))
                elif int(rank) <= threshold_rank:
                    if rank_results[f'{qid}_{pid}'] < threshold_score:
                        if qid not in top_results:
                            top_results[str(qid)] = []
                        top_results[str(qid)].append(str(pid))
        return retrieval_results, top_results
    
    @staticmethod
    def _read_rank_result(path):
        rank_results = {}
        with open(path) as f:
            for line in tqdm(f):
                qid, pid, score = line.rstrip().split()
                rank_results[f'{qid}_{pid}'] = float(score)
        return rank_results

    def __call__(self, example):
        query_id = example['query_id']
        if query_id in self.retrieval_results:
            # positive_ids = [pos['docid'] for pos in example['positive_passages']]
            hard_negatives = []
            if query_id in self.top_results:
                ranked_docid = self.top_results[query_id]
                for docid in np.random.choice(ranked_docid, min(self.sample_top, len(ranked_docid)), replace=False):
                    doc_info = self.corpus_data[self.docid_to_idx[docid]]
                    text = doc_info['text']
                    title = doc_info['title'] if 'title' in doc_info else None
                    if docid not in query2qrel[query_id]:    # positive_ids:
                        hn_doc = {'docid': docid, 'text': text}
                        if title:
                            hn_doc['title'] = title
                        hard_negatives.append(hn_doc)
            retrieved_docid = self.retrieval_results[query_id]
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
    parser.add_argument('--rank_result_path', type=str, required=True)
    parser.add_argument('--sample', type=int, default=100, required=True)
    parser.add_argument('--sample_top', type=int, default=10, required=True)
    parser.add_argument('--top', type=int, default=1000, required=True)
    parser.add_argument('--threshold_rank', type=int, default=0, required=True)
    parser.add_argument('--threshold_score', type=float, default=0.1, required=True)
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
        miner = BasicHardNegativeMiner(args.result_path, args.rank_result_path, corpus_data, args.sample, args.sample_top, args.top, args.threshold_rank, args.threshold_score)

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
