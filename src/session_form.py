import collections
import json
import numpy as np
import random
from tqdm import tqdm
import math
import os
import sys
import math



qid2qrel = []
for i, session in enumerate(['20', '40', '60', '80', '100']):
    tmp = collections.defaultdict(set)
    path = f'../data_incre4/msmarco-passage-{session}/test-qrels-{session}.tsv'
    with open(path, 'r') as f:
        for line in f:
            qid1, _, qid2, rel = line.strip().split()
            if int(rel) >= 2:
                tmp[int(qid1)].add(int(qid2))
    qid2qrel.append(tmp)
    print(len(tmp))


qid2qrel_session = []
for i in range(len(qid2qrel)):
    if i == 0:
        tmp = {qid:qid2qrel[i][qid] for qid in qid2qrel[i].keys()}
    else:
        tmp = {qid:qid2qrel[i][qid]-qid2qrel[i-1][qid] for qid in qid2qrel[i].keys()}
    qid2qrel_session.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))


print('session 1 ...')
qid2recall = collections.defaultdict(set)
with open('../zrun_er_20_e8_top1000_sample100/retrieval_msmarco_ckpt_20/rank.test.txt.trec', 'r') as f:
    for line in f:
        qid1, _, qid2, rank, _, _ = line.strip().split()
        qid2recall[int(qid1)].add(int(qid2))
qid2success = []
for qid2qrel_sess in qid2qrel_session:
    tmp = collections.defaultdict(set)
    for qid, qrel in qid2qrel_sess.items():
        tmp[qid] = set([did for did in qrel if did in qid2recall[qid]])
    qid2success.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))


print('session 2 ...')
qid2recall = collections.defaultdict(set)
with open('../zrun_mir_40_e8_top1000_sample100_n6+2_up10/retrieval_msmarco_ckpt_40/rank.test.txt.trec', 'r') as f:
    for line in f:
        qid1, _, qid2, rank, _, _ = line.strip().split()
        qid2recall[int(qid1)].add(int(qid2))
qid2success = []
for qid2qrel_sess in qid2qrel_session:
    tmp = collections.defaultdict(set)
    for qid, qrel in qid2qrel_sess.items():
        tmp[qid] = set([did for did in qrel if did in qid2recall[qid]])
    qid2success.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))


print('session 3 ...')
qid2recall = collections.defaultdict(set)
with open('../zrun_mir_3-5_e8_top1000_sample100_n6+2_up10/retrieval_msmarco_ckpt_60/rank.test.txt.trec', 'r') as f:
    for line in f:
        qid1, _, qid2, rank, _, _ = line.strip().split()
        qid2recall[int(qid1)].add(int(qid2))
qid2success = []
for qid2qrel_sess in qid2qrel_session:
    tmp = collections.defaultdict(set)
    for qid, qrel in qid2qrel_sess.items():
        tmp[qid] = set([did for did in qrel if did in qid2recall[qid]])
    qid2success.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))


print('session 4 ...')
qid2recall = collections.defaultdict(set)
with open('../zrun_mir_3-5_e8_top1000_sample100_n6+2_up10/retrieval_msmarco_ckpt_80/rank.test.txt.trec', 'r') as f:
    for line in f:
        qid1, _, qid2, rank, _, _ = line.strip().split()
        qid2recall[int(qid1)].add(int(qid2))
qid2success = []
for qid2qrel_sess in qid2qrel_session:
    tmp = collections.defaultdict(set)
    for qid, qrel in qid2qrel_sess.items():
        tmp[qid] = set([did for did in qrel if did in qid2recall[qid]])
    qid2success.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))


print('session 5 ...')
qid2recall = collections.defaultdict(set)
with open('../zrun_mir_3-5_e8_top1000_sample100_n6+2_up10/retrieval_msmarco_ckpt_100/rank.test.txt.trec', 'r') as f:
    for line in f:
        qid1, _, qid2, rank, _, _ = line.strip().split()
        qid2recall[int(qid1)].add(int(qid2))
qid2success = []
for qid2qrel_sess in qid2qrel_session:
    tmp = collections.defaultdict(set)
    for qid, qrel in qid2qrel_sess.items():
        tmp[qid] = set([did for did in qrel if did in qid2recall[qid]])
    qid2success.append(tmp)
    print(np.sum([len(val) for val in tmp.values()])/len(qid2qrel[-1].keys()))
