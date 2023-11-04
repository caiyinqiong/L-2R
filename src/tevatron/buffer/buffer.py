from .random_retrieve import Random_retrieve
from .reservoir_update import Reservoir_update
from .mir_retrieve import MIR_retrieve
from .gss_greedy_update import GSSGreedyUpdate

from .our_retrieve import Our_retrieve
from .our_update import OurUpdate

from .our_retrieve_emb import Our_retrieve_emb
from .our_update_emb import OurUpdateEmb

from .our_retrieve_emb_cosine import Our_retrieve_emb_cosine
from .our_update_emb_cosine import OurUpdateEmbCosine

from .our_retrieve_emb_vertical import Our_retrieve_emb_vertical
from .our_update_emb_vertical import OurUpdateEmbVertical

from .our_retrieve_emb_horizontal import Our_retrieve_emb_horizontal
from .our_update_emb_horizontal import OurUpdateEmbHorizontal

from tevatron.arguments import DataArguments, TevatronTrainingArguments

import torch
import pickle
import os
import collections
import json


retrieve_methods = {
    'random': Random_retrieve,
    'mir': MIR_retrieve,
    'our': Our_retrieve,
    # 'our_emb': Our_retrieve_emb,
    'our_emb_cosine': Our_retrieve_emb_cosine,
    'our_emb_vertical': Our_retrieve_emb_vertical,
    'our_emb_horizontal': Our_retrieve_emb_horizontal,
}
update_methods = {
    'random': Reservoir_update,
    'gss': GSSGreedyUpdate,
    'our': OurUpdate,
    # 'our_emb': OurUpdateEmb,
    'our_emb_cosine': OurUpdateEmbCosine,
    'our_emb_vertical': OurUpdateEmbVertical,
    'our_emb_horizontal': OurUpdateEmbHorizontal,
}


class Buffer(torch.nn.Module):
    def __init__(self, model, tokenizer, params: DataArguments, train_params: TevatronTrainingArguments):
        super().__init__()
        self.params = params
        self.train_params = train_params
        self.model = model
        self.tokenizer = tokenizer
        self.buffer_size = params.mem_size
        self.n_seen_so_far = collections.defaultdict(int)  # 目前已经过了多少个样本了, 只有er需要
        self.buffer_qid2dids = collections.defaultdict(list)

        if self.params.update_method == 'gss':
            buffer_score = None

        if params.buffer_data:
            print('load buffer data from %s' % params.buffer_data)
            pkl_file = open(os.path.join(params.buffer_data, 'buffer.pkl'), 'rb')
            if self.params.update_method == 'gss':
                self.n_seen_so_far, self.buffer_qid2dids, buffer_score = pickle.load(pkl_file)
            else:
                self.n_seen_so_far, self.buffer_qid2dids = pickle.load(pkl_file)
            pkl_file.close()
        
            if params.compatible:
                pkl_file = open(os.path.join(params.buffer_data, 'buffer_emb.pkl'), 'rb')
                self.buffer_did2emb = pickle.load(pkl_file)
                pkl_file.close()

        self.qid2query = self.read_data(is_query=True, data_path=params.query_data)   # {'qid':query text}
        self.did2doc = self.read_data(is_query=False, data_path=params.doc_data)   # {'docid':doc text}
        print('total did2doc:', len(self.did2doc))

        if self.params.update_method == 'gss':
            self.update_method = update_methods[params.update_method](params, train_params, buffer_score=buffer_score)
        else:
            self.update_method = update_methods[params.update_method](params, train_params)
        self.retrieve_method = retrieve_methods[params.retrieve_method](params, train_params)

    def read_data(self, is_query, data_path):
        print('load data from %s' % data_path)
        id2text = {}
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if is_query:
                    id2text[data['query_id']] = data['query']
                else:
                    id2text[data['docid']] = data['title'] + self.params.passage_field_separator + data['text'] if 'title' in data else data['text']
        return id2text

    def update(self, qid_lst, docids_lst, **kwargs):
        return self.update_method.update(buffer=self, qid_lst=qid_lst, docids_lst=docids_lst, **kwargs)

    def retrieve(self, qid_lst, docids_lst, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, qid_lst=qid_lst, docids_lst=docids_lst, **kwargs)
    
    def replace(self, **kwargs):
        return self.update_method.replace(buffer=self, **kwargs)

    def save(self, output_dir: str):
        output = open(os.path.join(output_dir, 'buffer.pkl'), 'wb')
        if self.params.update_method == 'gss':
            pickle.dump((self.n_seen_so_far, self.buffer_qid2dids, self.update_method.buffer_score), output)
        else:
            pickle.dump((self.n_seen_so_far, self.buffer_qid2dids), output)
        output.close()
