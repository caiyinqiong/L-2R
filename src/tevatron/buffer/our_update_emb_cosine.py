import torch
import torch.nn.functional as F
import collections
import copy
import numpy as np
from tqdm import tqdm

from .buffer_utils import get_grad_vector, cosine_similarity, random_retrieve


class OurUpdateEmbCosine(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params

        self.mem_eval_size = params.mem_eval_size
        self.mem_replace_size = params.mem_replace_size
        self.upsample_scale = params.upsample_scale

        self.candidate_neg_docids = collections.defaultdict(set)

    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        candidate_neg_docids = kwargs['candidate_neg_docids']
        for qid, docids in candidate_neg_docids.items():
            self.candidate_neg_docids[qid].update(docids)
    
    def replace(self, buffer):
        model_temp = copy.deepcopy(buffer.model)
        model_temp.eval()
        
        for qid, candidate_docids in tqdm(self.candidate_neg_docids.items(), total=len(self.candidate_neg_docids)):
            candidate_docids = np.array(list(candidate_docids))

            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left == 0:  # buffer满， 执行替换
                mem_eval_size = min(self.mem_eval_size, len(buffer.buffer_qid2dids[qid]))
                mem_eval_docids_lst, mem_eval_indices = random_retrieve(buffer.buffer_qid2dids[qid], mem_eval_size, return_indices=True)
                # mem_eval_model_out = self.get_model_out(buffer, model_temp, qid, mem_eval_docids_lst)  # [mem_eval_size, 768]

                mem_upsample_num = min(len(buffer.buffer_qid2dids[qid])-mem_eval_size, int(self.mem_replace_size*self.upsample_scale))  # 上采样mem数据个数
                upsample_mem_docids_lst, upsample_mem_indices = random_retrieve(buffer.buffer_qid2dids[qid], mem_upsample_num, excl_indices=mem_eval_indices, return_indices=True)
                # mem_model_out = self.get_model_out(buffer, model_temp, qid, upsample_mem_docids_lst)

                new_upsample_num = min(len(candidate_docids), int(self.mem_replace_size*self.upsample_scale))  # 上采样新数据个数
                upsample_candidate_docids_lst = random_retrieve(candidate_docids, new_upsample_num)
                # new_model_out = self.get_model_out(buffer, model_temp, qid, upsample_candidate_docids_lst)

                if self.params.compatible:
                    mem_eval_model_out = torch.tensor(np.array([buffer.buffer_did2emb[int(docid)] for docid in mem_eval_docids_lst]), device=self.train_params.device)
                    mem_model_out = torch.tensor(np.array([buffer.buffer_did2emb[int(docid)] for docid in upsample_mem_docids_lst]), device=self.train_params.device)
                    new_model_out = self.get_model_out(buffer, model_temp, qid, upsample_candidate_docids_lst)
                else:
                    model_out = self.get_model_out(buffer, model_temp, qid, mem_eval_docids_lst+upsample_mem_docids_lst+upsample_candidate_docids_lst)
                    mem_eval_model_out = model_out[:mem_eval_size,:]  # [mem_eval_size, 768]
                    mem_model_out = model_out[mem_eval_size:mem_eval_size+mem_upsample_num,:]
                    new_model_out = model_out[mem_eval_size+mem_upsample_num:,:]
                
                mem_sim = cosine_similarity(mem_model_out, mem_eval_model_out)  # [mem_upsample_num, mem_eval_size]
                mem_sim = torch.sum(mem_sim, dim=-1) / mem_sim.size(-1)  # 选相似度最大的一些
                indices = mem_sim.sort(dim=0, descending=True)[1]

                new_sim = cosine_similarity(new_model_out, mem_eval_model_out)  # [new_upsample_num, mem_eval_size]
                new_sim = torch.sum(new_sim, dim=-1) * (-1.0) / new_sim.size(-1)  # 选相似度最小的一些
                new_indices = new_sim.sort(dim=0, descending=True)[1]

                mem_replace_size = min([self.mem_replace_size, len(mem_sim), len(new_sim)])
                indices = indices[:mem_replace_size]
                new_indices = new_indices[:mem_replace_size]
                # 执行替换
                buffer.buffer_qid2dids[qid] = np.array(buffer.buffer_qid2dids[qid])
                buffer.buffer_qid2dids[qid][np.array(upsample_mem_indices)[indices.cpu()]] = np.array(upsample_candidate_docids_lst)[new_indices.cpu()].copy()
                buffer.buffer_qid2dids[qid] = buffer.buffer_qid2dids[qid].tolist()
            
            elif place_left == buffer.buffer_size:  # buffer 全空，随机选择新数据放入
                all_indices = np.arange(len(candidate_docids))
                num_retrieve = min(len(candidate_docids), buffer.buffer_size)
                indices = list(np.random.choice(all_indices, num_retrieve, replace=False))
                buffer.buffer_qid2dids[qid].extend(candidate_docids[indices])

            else:  # buffer 有剩余，选择与eval mem最不接近的place_left个新数据，放入
                mem_eval_size = min(self.mem_eval_size, len(buffer.buffer_qid2dids[qid]))
                mem_eval_docids_lst = random_retrieve(buffer.buffer_qid2dids[qid], mem_eval_size)

                upsample_num = min(len(candidate_docids), int(place_left*self.upsample_scale))  # 两倍量的采样
                upsample_candidate_docids_lst = random_retrieve(candidate_docids, upsample_num)  # list

                if self.params.compatible:
                    mem_eval_model_out = torch.tensor(np.array([buffer.buffer_did2emb[int(docid)] for docid in mem_eval_docids_lst]), device=self.train_params.device)
                    candidate_model_out = self.get_model_out(buffer, model_temp, qid, upsample_candidate_docids_lst)  # [upsample_num, 768]
                else:
                    model_out = self.get_model_out(buffer, model_temp, qid, mem_eval_docids_lst+upsample_candidate_docids_lst)  # [mem_eval_size+upsample_num, 768]
                    mem_eval_model_out = model_out[:mem_eval_size,:]
                    candidate_model_out = model_out[mem_eval_size:,:]

                inter_sim = cosine_similarity(candidate_model_out, mem_eval_model_out)  # [upsample_num, mem_eval_size]
                inter_sim = torch.sum(inter_sim, dim=-1) * (-1.0) / inter_sim.size(-1)  # [upsample_num] 选相似度最小的一些

                num_new = min(place_left, len(candidate_model_out))
                indices = inter_sim.sort(dim=0, descending=True)[1][:num_new]
                buffer.buffer_qid2dids[qid].extend(np.array(upsample_candidate_docids_lst)[indices.cpu()])

    def get_model_out(self, buffer, model_temp, qid, docids_lst):
        doc_lst = [buffer.did2doc[did] for did in docids_lst]
        doc_lst = buffer.tokenizer.batch_encode_plus(doc_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        for key, value in doc_lst.items():
            doc_lst[key] = value.to(self.train_params.device)
        p_reps = model_temp(None, doc_lst).p_reps
        return p_reps
