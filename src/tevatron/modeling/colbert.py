import torch
import torch.nn as nn
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel, EncoderOutput
from transformers import AutoTokenizer
import string
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ColbertPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, tied=True):
        super(ColbertPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class ColbertModel(EncoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.skiplist = self.tokenizer.encode(string.punctuation, add_special_tokens=False)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.pooler(p=p_hidden)
        mask = torch.tensor(self.mask(psg['input_ids']), device=psg['input_ids'].device)
        p_reps *= mask[:, :, None].float()
        return p_reps, mask

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.pooler(q=q_hidden)
        q_reps *= qry['attention_mask'][:, :, None].float()
        return q_reps, qry['attention_mask']

    def compute_similarity(self, q_reps, p_reps):
        if self.training:
            token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
            scores, _ = token_scores.max(-1)
            scores = scores.sum(1)
        else:
            scores = (q_reps @ p_reps.permute(0, 2, 1)).max(dim=2).values.sum(dim=1)
        return scores

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps, q_mask = self.encode_query(query)
        p_reps, p_mask = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            if q_reps is not None:  # query
                q_mask = q_mask.bool()
                q_reps = [emb[q_mask[idx]] for idx, emb in enumerate(q_reps)]
                q_len = [m.sum() for m in q_mask]
                return EncoderOutput(
                    q_reps=q_reps,
                    p_reps=p_reps,
                    scores=q_len
                )
            if p_reps is not None:  # passage
                p_mask = p_mask.bool()
                p_reps = [emb[p_mask[idx]] for idx, emb in enumerate(p_reps)]
                p_len = [m.sum() for m in p_mask]
                return EncoderOutput(
                    q_reps=q_reps,
                    p_reps=p_reps,
                    scores=p_len
                )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
                
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ColbertPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = ColbertPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
