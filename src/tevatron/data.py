import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry_id = group['query_id']
        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        psg_ids = []
        encoded_passages = []
        group_positives = [(docid, pos) for docid, pos in zip(group['pos_docids'], group['positives'])]
        group_negatives = [(docid, neg) for docid, neg in zip(group['neg_docids'], group['negatives'])]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        psg_ids.append(pos_psg[0])
        encoded_passages.append(self.create_one_example(pos_psg[1]))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            psg_ids.append(neg_psg[0])
            encoded_passages.append(self.create_one_example(neg_psg[1]))

        return qry_id, psg_ids, encoded_query, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


class RankDataset(Dataset):
    input_keys =  ['query_id', 'query', 'doc_id', 'doc']

    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer):
        self.rank_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.rank_data)

    def __len__(self):
        return len(self.rank_data)
    
    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        data = self.rank_data[item]
        qid = data['query_id']
        did = data['doc_id']
        encoded_query = self.create_one_example(data['query'], is_query=True)
        encoded_passage = self.create_one_example(data['doc'], is_query=False)

        return qid, did, encoded_query, encoded_passage


class RerankDataset(Dataset):
    input_keys =  ['query_id', 'query', 'doc_id', 'doc']

    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer):
        self.rank_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.rank_data)

    def __len__(self):
        return len(self.rank_data)
    
    def create_one_example(self, query_encoding: List[int], doc_encoding: List[int]):
        item = self.tok.encode_plus(
            text=query_encoding,
            text_pair=doc_encoding,
            truncation='only_first',
            max_length=self.data_args.seq_max_len,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        data = self.rank_data[item]
        qid = data['query_id']
        did = data['doc_id']
        encoded_qp = self.create_one_example(data['query'], data['doc'])

        return qid, did, encoded_qp


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq_id = [f[0] for f in features]
        dd_id = [f[1] for f in features]
        qq = [f[2] for f in features]
        dd = [f[3] for f in features]

        if isinstance(qq_id[0], list):
            qq_id = sum(qq_id, [])
        if isinstance(dd_id[0], list):
            dd_id = sum(dd_id, [])
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return qq_id, dd_id, q_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features


@dataclass
class RankCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq_id = [f[0] for f in features]
        dd_id = [f[1] for f in features]
        qq = [f[2] for f in features]
        dd = [f[3] for f in features]

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return qq_id, dd_id, q_collated, d_collated


@dataclass
class RerankCollator(DataCollatorWithPadding):
    max_seq_len: int = 160

    def __call__(self, features):
        qq_id = [f[0] for f in features]
        dd_id = [f[1] for f in features]
        qd = [f[2] for f in features]

        qd_collated = self.tokenizer.pad(
            qd,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return qq_id, dd_id, qd_collated
