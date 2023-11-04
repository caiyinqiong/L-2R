import collections
import json
import numpy as np
import random
from tqdm import tqdm
import math
import os
import sys
import math


def Recall(qrels, pred, threshold):
    if len(qrels) == 0:
        return 0.0
    if threshold == -1:
        pred = pred
    else: 
        pred = pred[:threshold]

    score = 0.0
    for (item, label) in qrels:
        if item in set(pred):
            score += 1.0
    return score / len(qrels)


def Precision(qrels, pred, threshold):
    if len(qrels) == 0:
        return 0.0
    if threshold == -1:
        pred = pred
    else: 
        pred = pred[:threshold]
    
    score = 0.0
    for (item, label) in qrels:
        if item in set(pred):
            score += 1.0
    return score / len(pred)


def DCG(qrels, pred, threshold):
    if len(qrels) == 0:
        return 0.0
    qrels = {did:label for (did, label) in qrels} 

    score = 0.0
    for i, did in enumerate(pred[:threshold]):
        if i == 0:
            if did in qrels:
                score += qrels[did]
        else:
            if did in qrels:
                score += qrels[did] / math.log2(2. + i)
    return score


def NDCG(qrels, pred, threshold):
    if len(qrels) == 0:
        return 0.0
    
    dcg = DCG(qrels, pred, threshold)
    
    ground = sorted(qrels, key=lambda x:x[1], reverse=True)
    ground = [did for (did, _) in ground]
    idcg = DCG(qrels, ground, threshold)
    return dcg / idcg


def cal_metrics(qrels_file, pred_file, type, thre):
    test_qids = set()
    test_qid2qrel = collections.defaultdict(list)
    with open(qrels_file) as f:
        for _, line in enumerate(f):
            qid1, _, qid2, rel = line.strip().split()
            test_qids.add(int(qid1))
            if int(rel) >= thre:
                if type == 'doc':
                    qid1, qid2, rel = int(qid1), int(qid2[1:]), int(rel)
                elif type == 'passage':
                    qid1, qid2, rel = int(qid1), int(qid2), int(rel)
                test_qid2qrel[qid1].append((qid2, rel))
    print('total test query number', len(test_qids))
    print('test avg pos sample number', len(test_qid2qrel), np.mean([len(qrel) for qrel in test_qid2qrel.values()]))

    test_qid2pred = collections.defaultdict(list)
    with open(pred_file) as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            qid1, _, qid2, rank, _, _ = line.strip().split()   ##########################
            # qid1, qid2, rank = line.strip().split() 
            if type == 'doc':
                qid1, qid2, rank = int(qid1), int(qid2[1:]), int(rank)
            elif type == 'passage':
                qid1, qid2, rank = int(qid1), int(qid2), int(rank)
            test_qid2pred[qid1].append((qid2, rank))
    print(len(test_qid2pred), np.mean([len(pred) for pred in test_qid2pred.values()]))
    metric = collections.defaultdict(list)
    for qid in tqdm(test_qids, total=len(test_qids)):
        qrels = test_qid2qrel[qid]
        pred_out = test_qid2pred[qid]
        pred_out.sort(key=lambda x:x[1])
        pred = [qid2 for (qid2, _) in pred_out]

        # metric['Recall@100'].append(Recall(qrels, pred, 100))
        # metric['Recall@1000'].append(Recall(qrels, pred, 1000))

        metric['NDCG@10'].append(NDCG(qrels, pred, 10))
        metric['NDCG@100'].append(NDCG(qrels, pred, 100))

        metric['DCG@10'].append(DCG(qrels, pred, 10))
        metric['DCG@100'].append(DCG(qrels, pred, 100))

        # metric['Precision@10'].append(Precision(qrels, pred, 10))
        # metric['Precision@100'].append(Precision(qrels, pred, 100))

    for key, val in metric.items():
        print(key, np.mean(val))


def main():
    print("Eval Started")
    cal_metrics(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    

if __name__ == '__main__':
    main()


# python XXX.py <reference ranking> <candidate ranking> passage 2
