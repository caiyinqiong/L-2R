from .buffer_utils import random_retrieve
import torch
import numpy as np


class Random_retrieve(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.num_retrieve = params.mem_batch_size  # memory data要使用的 batch size

    def retrieve(self, buffer, qid_lst, docids_lst, **kwargs):
        num_retrieve = self.num_retrieve
        for qid in qid_lst:
            num_retrieve = min(num_retrieve, len(buffer.buffer_qid2dids[qid]))
        if num_retrieve == 0:
            return None

        docids_lst_from_mem = []   # [num_q * mem_batch_size]
        for qid in qid_lst:
            docids = random_retrieve(buffer.buffer_qid2dids[qid], num_retrieve)
            docids_lst_from_mem.extend(docids)

        doc_lst_from_mem = [buffer.did2doc[did] for did in docids_lst_from_mem]
        doc_lst_from_mem = buffer.tokenizer.batch_encode_plus(doc_lst_from_mem,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.params.p_max_len,
                    truncation='only_first',
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt")
        
        if self.params.compatible:
            return torch.tensor(docids_lst_from_mem), doc_lst_from_mem  # [num_q * mem_batch_size], cpu; [num_q * mem_batch_size, doc_len], cpu

        return doc_lst_from_mem   # [num_q * mem_batch_size, doc_len], cpu
