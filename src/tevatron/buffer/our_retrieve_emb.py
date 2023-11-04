import torch
import torch.nn.functional as F
from .buffer_utils import random_retrieve, get_grad_vector, cosine_similarity, cosine_similarity_3d
import copy
import numpy as np
import collections


class Our_retrieve_emb(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params
        
        self.alpha = params.alpha
        self.beta = params.beta
        self.new_bz = params.new_batch_size

        self.mem_upsample = params.mem_upsample
        self.mem_bz = params.mem_batch_size

    def retrieve(self, buffer, qid_lst, docids_lst, **kwargs):
        model_temp = copy.deepcopy(buffer.model)
        model_temp.eval()
        
        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,:1]  # pos passage
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,1:]  # 去掉pos passage
        q_lst, d_lst = kwargs['q_lst'], kwargs['d_lst']

        res_d_lst = collections.defaultdict(list)
        res_neg_did_lst = collections.defaultdict(set)

        ############## 处理new data #############
        new_model_out = model_temp(q_lst, d_lst)
        index_new = self.get_new_data(new_model_out, self.new_bz, self.alpha, self.beta)  # 得到选择出来的新数据的负例下标[batch_size, new_bz], cuda
        for i, qid in enumerate(qid_lst):
            res_neg_did_lst[qid].update(docids_neg_lst[i][index_new[i].cpu()])  # 选择出来的负例
        index_new = torch.cat((torch.zeros_like(index_new[:,:1]), index_new+1), dim=-1)  # [batch_size, new_bz+1]
        for key, val in d_lst.items():
            val = val.reshape(batch_size, -1, val.size(-1))
            res_d_lst[key].append(torch.gather(val, 1, index_new.unsqueeze(dim=2).repeat(1,1,val.size(-1))))  # [batch_size, new_bz+1, 128]
        
        ############### 处理mem data ##############
        buffer_len = min([len(buffer.buffer_qid2dids[qid]) for qid in qid_lst])
        mem_upsample = min(self.mem_upsample, buffer_len)
        mem_bz = min(mem_upsample, self.mem_bz)
        if mem_upsample > 0 and mem_bz > 0:
            mem_upsample_docids_lst = []
            for i, qid in enumerate(qid_lst):
                mem_upsample_docids = random_retrieve(buffer.buffer_qid2dids[qid], mem_upsample)
                mem_upsample_docids_lst.extend(docids_pos_lst[i].tolist() + mem_upsample_docids)  # 把正例也加进去
            mem_doc_lst = [buffer.did2doc[did] for did in mem_upsample_docids_lst]
            mem_doc_lst = buffer.tokenizer.batch_encode_plus(mem_doc_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
            for key, value in mem_doc_lst.items():
                mem_doc_lst[key] = value.to(self.train_params.device)  # mem_doc_lst = [batch_size*(1+mem_upsample), 128]
            mem_model_out = model_temp(q_lst, mem_doc_lst)
            index_mem = self.get_mem_data(new_model_out, index_new, mem_model_out, mem_bz)  # [batch_size, mem_bz]
            index_mem = index_mem + 1  # [batch_size, mem_bz]
            for key, val in mem_doc_lst.items():
                val = val.reshape(batch_size, -1, val.size(-1))  # # [batch_size, mem_upsample, 128]
                res_d_lst[key].append(torch.gather(val, 1, index_mem.unsqueeze(dim=2).repeat(1,1,val.size(-1))))  # [batch_size, mem_bz, 128]

        for key, val in res_d_lst.items():
            val = torch.cat(val, dim=1)  # [batch_size, new_bz+mem_bz+1, 128]
            res_d_lst[key] = val.reshape(-1, val.size(-1))  # [batch_size*(new_bz+mem_bz+1), 128]
        return res_d_lst, None, res_neg_did_lst

    def get_mem_data(self, new_model_out, index_new, mem_model_out, mem_bz):
        new_q_reps = new_model_out.q_reps
        new_p_reps = new_model_out.p_reps
        new_p_reps = new_p_reps.reshape(new_q_reps.size(0), -1, new_p_reps.size(1))  # [bz, n, 768]
        choiced_new_reps = torch.gather(new_p_reps, 1, index_new.unsqueeze(dim=2).repeat(1,1,new_p_reps.size(-1)))[:,1:,:]  # [batch_size, new_bz, 768]

        p_reps = mem_model_out.p_reps  # [bz*n, 768]
        p_reps = p_reps.reshape(new_q_reps.size(0), -1, p_reps.size(1))  # [bz, n, 768]
        neg_p_reps = p_reps[:,1:,:]  # [bz, n-1, 768]

        inter_sim = cosine_similarity_3d(neg_p_reps, choiced_new_reps)  # [bz, n-1, new_bz]
        inter_sim_sum = torch.sum(inter_sim, dim=-1)  # [bz, n-1]
        inter_sim = inter_sim_sum * (-1.0) / inter_sim.size(-1)  # [bz, n-1], 相似度尽可能小

        indexs = inter_sim.sort(dim=1, descending=True)[1][:,:mem_bz]
        return indexs

    def get_new_data(self, new_model_out, new_bz, alpha, beta):
        q_reps = new_model_out.q_reps  # [8, 768]
        p_reps = new_model_out.p_reps  # [8*n, 768]
        p_reps = p_reps.reshape(q_reps.size(0), -1, p_reps.size(1))  # [8, n, 768]

        q_reps_norm = q_reps.norm(p=2, dim=1, keepdim=True)  # [8, 1]
        q_reps = q_reps.unsqueeze(dim=1)  # [8, 1, 768]
        p_q = torch.matmul(p_reps, q_reps.transpose(1, 2)) * q_reps / (q_reps_norm * q_reps_norm).unsqueeze(dim=2).clamp(min=1e-8)  # p在q方向上的投影向量, [8, n, 768]
        neg = p_q[:,1:,:]  # [8, n-1, 768]
        pos= p_q[:,:1,:].repeat(1, neg.size(1), 1)  # [8, n-1, 768]
        dis = F.pairwise_distance(neg.reshape(-1, neg.size(-1)), pos.reshape(-1, pos.size(-1)), p=2.0).reshape(neg.size(0), -1)  # [8, n-1], 尽可能大 

        neg_p_reps = p_reps[:,1:,:]  # [8, n-1, 768]
        inter_sim = cosine_similarity_3d(neg_p_reps, neg_p_reps)  # [8, n-1, n-1]
        inter_sim_sum = torch.sum(inter_sim, dim=-1)  # [8, n-1]
        inter_sim = (inter_sim_sum - torch.ones_like(inter_sim_sum)) / (inter_sim.size(-1) - 1)  # [8, n-1], 尽可能小

        sim = alpha * dis - beta * inter_sim
        indexs = sim.sort(dim=1, descending=True)[1][:,:new_bz]
        return indexs
