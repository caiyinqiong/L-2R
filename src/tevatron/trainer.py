import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np

from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from tevatron.buffer.buffer import Buffer

import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class TevatronTrainer(Trainer):
    def __init__(self, data_args, tokenizer, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        if self.data_args.cl_method:
            self.buffer = Buffer(self.model, self.tokenizer, self.data_args, self.args)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        if self.data_args.cl_method == 'our' and self.state.global_step == self.state.max_steps:
            print('update buffer...')
            self.buffer.replace()
        if self.data_args.cl_method:
            self.buffer.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs[2:]:
            for key, val in x.items():
                x[key] = val.to(self.args.device)
            prepared.append(x)

        if self.data_args.cl_method == 'er':
            if not self.data_args.compatible:
                qid_lst, docids_lst = inputs[0], inputs[1]

                mem_passage = self.buffer.retrieve(
                    qid_lst=qid_lst, docids_lst=docids_lst, 
                    q_lst=prepared[0], d_lst=prepared[1], lr=self._get_learning_rate()
                )  # ER: [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz, d_len], gpu
                self.buffer.update(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    q_lst=prepared[0], d_lst=prepared[1]
                )

                if mem_passage is not None:
                    for key, val in mem_passage.items():
                        passage_len = val.size(-1)
                        prepared[1][key] = prepared[1][key].reshape(len(qid_lst), -1, passage_len)  # [num_q, bz, d_len]
                        val = val.reshape(len(qid_lst), -1, passage_len).to(prepared[1][key].device)  # [num_q, mem_bz, d_len]
                        prepared[1][key] = torch.cat((prepared[1][key], val), dim=1).reshape(-1, passage_len)  # [num_q*(bz+mem_bz), d_len]
            else:
                qid_lst, docids_lst = inputs[0], inputs[1]

                mem_docids_lst, mem_passage = self.buffer.retrieve(
                    qid_lst=qid_lst, docids_lst=docids_lst, 
                    q_lst=prepared[0], d_lst=prepared[1], lr=self._get_learning_rate()
                )  # ER:[num_q * mem_bz],cpu, [num_q * mem_bz, d_len], cpu; MIR: MIR: [num_q, mem_bz],cpu, [num_q, mem_bz, d_len], gpu
                self.buffer.update(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    q_lst=prepared[0], d_lst=prepared[1]
                )

                if mem_passage is not None:
                    for key, val in mem_passage.items():
                        passage_len = val.size(-1)
                        prepared[1][key] = prepared[1][key].reshape(len(qid_lst), -1, passage_len)  # [num_q, bz, d_len]
                        val = val.reshape(len(qid_lst), -1, passage_len).to(prepared[1][key].device)  # [num_q, mem_bz, d_len]
                        prepared[1][key] = torch.cat((prepared[1][key], val), dim=1).reshape(-1, passage_len)  # [num_q*(bz+mem_bz), d_len]
                    
                    docids_lst = torch.tensor(docids_lst).reshape(len(qid_lst), -1)  # [num_q, n]
                    mem_docids_lst = mem_docids_lst.reshape(len(qid_lst), -1)  # [num_q, mem_bz]
                    all_docids_lst = torch.cat((docids_lst, mem_docids_lst), dim=-1).reshape(-1)  # [num_q * n+mem_bz]

                    identity = []
                    doc_oldemb = []
                    for i, docids in enumerate(all_docids_lst):
                        docids = int(docids)
                        if docids in self.buffer.buffer_did2emb:
                            identity.append(i)
                            doc_oldemb.append(self.buffer.buffer_did2emb[docids])
                    identity = torch.tensor(identity)
                    doc_oldemb = torch.tensor(np.array(doc_oldemb), device=self.args.device)
                    prepared.append(identity)
                    prepared.append(doc_oldemb)
        elif self.data_args.cl_method == 'our':
            if not self.data_args.compatible:
                qid_lst, docids_lst = inputs[0], inputs[1]

                mem_passage, pos_docids, candidate_neg_docids = self.buffer.retrieve(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    q_lst=prepared[0], d_lst=prepared[1]
                )    # [num_q*(new_bz+mem_bz), d_len]
                self.buffer.update(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    pos_docids=pos_docids, candidate_neg_docids=candidate_neg_docids
                )

                if mem_passage is not None:
                    for key, val in mem_passage.items():
                        prepared[1][key] = val
            else:
                qid_lst, docids_lst = inputs[0], inputs[1]

                mem_emb, mem_passage, pos_docids, candidate_neg_docids = self.buffer.retrieve(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    q_lst=prepared[0], d_lst=prepared[1]
                )    # [num_q*(1+mem_bz), 768],gpu; [num_q*(new_bz+mem_bz), d_len],gpu
                self.buffer.update(
                    qid_lst=qid_lst, docids_lst=docids_lst,
                    pos_docids=pos_docids, candidate_neg_docids=candidate_neg_docids
                )

                if mem_passage is not None:
                    for key, val in mem_passage.items():
                        prepared[1][key] = val
                    
                    identity = []  # [1+mem_batch_size, num_q]
                    pos_identity = torch.arange(len(qid_lst)) * (1 + self.data_args.new_batch_size + self.data_args.mem_batch_size)
                    identity.append(pos_identity)
                    for i in range(self.data_args.mem_batch_size):
                        identity.append(pos_identity + i + 1 + self.data_args.new_batch_size)
                    identity = torch.stack(identity, dim=0).transpose(0,1).reshape(-1)
                    prepared.append(identity)

                    prepared.append(mem_emb)
        elif self.data_args.cl_method == 'incre':
            if self.data_args.compatible:
                qid_lst, docids_lst = inputs[0], inputs[1]
                docids_lst = torch.tensor(docids_lst).reshape(len(qid_lst), -1)  # [num_q, n]

                identity = torch.arange(docids_lst.size(0)) * docids_lst.size(1)
                prepared.append(identity)

                doc_oldemb = []  # [num_q, 768]
                for docid in docids_lst[:,0]:  # 对于incre，只有正例是old doc
                    doc_oldemb.append(self.buffer.buffer_did2emb[int(docid)])
                doc_oldemb = torch.tensor(np.array(doc_oldemb)).to(self.args.device)
                prepared.append(doc_oldemb)
        else:
            print('not implement...')

        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage, identity, oldemb = inputs
        return model(query=query, passage=passage, identity=identity, oldemb=oldemb).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
