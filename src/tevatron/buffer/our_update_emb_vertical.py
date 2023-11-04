import torch
import torch.nn.functional as F
import collections
import copy
import numpy as np
from tqdm import tqdm

from .buffer_utils import get_grad_vector, cosine_similarity, random_retrieve


# 与query垂直方向
class OurUpdateEmbVertical(object):
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

                mem_upsample_num = min(len(buffer.buffer_qid2dids[qid])-mem_eval_size, int(self.mem_replace_size*self.upsample_scale))  # 上采样mem数据个数
                upsample_mem_docids_lst, upsample_mem_indices = random_retrieve(buffer.buffer_qid2dids[qid], mem_upsample_num, excl_indices=mem_eval_indices, return_indices=True)

                new_upsample_num = min(len(candidate_docids), int(self.mem_replace_size*self.upsample_scale))  # 上采样新数据个数
                upsample_candidate_docids_lst = random_retrieve(candidate_docids, new_upsample_num)

                q_reps, p_reps = self.get_model_out(buffer, model_temp, qid, mem_eval_docids_lst+upsample_mem_docids_lst+upsample_candidate_docids_lst)
                # p在q方向上的投影向量
                q_reps_norm = q_reps.norm(p=2, dim=-1, keepdim=True)  # [1, 1]
                p_q = torch.matmul(p_reps, q_reps.transpose(0, 1)) * q_reps / (q_reps_norm * q_reps_norm).clamp(min=1e-8)  # p在q方向上的投影向量, [mem_eval_size+mem_upsample_num+new_upsample_num, 768]
                # p在q垂直方向上的向量
                p_q_vertical = p_reps - p_q   # [mem_eval_size+mem_upsample_num+new_upsample_num, 768]
                mem_eval_p_q_vertical = p_q_vertical[:mem_eval_size,:]  # [mem_eval_size, 768]
                mem_p_q_vertical = p_q_vertical[mem_eval_size:mem_eval_size+mem_upsample_num,:]  # [mem_upsample_num, 768]
                new_p_q_vertical = p_q_vertical[mem_eval_size+mem_upsample_num:,:]  # [new_upsample_num, 768]

                memeval_p_q_vertical = mem_eval_p_q_vertical.unsqueeze(dim=0).repeat(mem_upsample_num,1,1)  # [mem_upsample_num, mem_eval_size, 768]
                mem_p_q_vertical = mem_p_q_vertical.unsqueeze(dim=1).repeat(1,mem_eval_size,1)  # [mem_upsample_num, mem_eval_size, 768]
                mem_inter_dis = F.pairwise_distance(mem_p_q_vertical.reshape(-1, mem_p_q_vertical.size(-1)), memeval_p_q_vertical.reshape(-1, memeval_p_q_vertical.size(-1)), p=2.0)  # [mem_upsample_num*mem_eval_size]
                mem_inter_dis = mem_inter_dis.reshape(mem_p_q_vertical.size(0), mem_p_q_vertical.size(1))  # [mem_upsample_num, mem_eval_size]
                mem_inter_dis = torch.sum(mem_inter_dis, dim=-1) / mem_inter_dis.size(-1)  # [upsample_num], 选尽可能小的
                indices = mem_inter_dis.sort(dim=0, descending=False)[1]

                neweval_p_q_vertical = mem_eval_p_q_vertical.unsqueeze(dim=0).repeat(new_upsample_num,1,1)  # [new_upsample_num, mem_eval_size, 768]
                new_p_q_vertical = new_p_q_vertical.unsqueeze(dim=1).repeat(1,mem_eval_size,1)  # [new_upsample_num, mem_eval_size, 768]
                new_inter_dis = F.pairwise_distance(new_p_q_vertical.reshape(-1, new_p_q_vertical.size(-1)), neweval_p_q_vertical.reshape(-1, neweval_p_q_vertical.size(-1)), p=2.0)  # [new_upsample_num*mem_eval_size]
                new_inter_dis = new_inter_dis.reshape(new_p_q_vertical.size(0), new_p_q_vertical.size(1))  # [mem_upsample_num, mem_eval_size]
                new_inter_dis = torch.sum(new_inter_dis, dim=-1) / new_inter_dis.size(-1)  # [upsample_num], 尽可能大
                new_indices = new_inter_dis.sort(dim=0, descending=True)[1]

                mem_replace_size = min([self.mem_replace_size, mem_upsample_num, new_upsample_num])
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

                q_reps, p_reps = self.get_model_out(buffer, model_temp, qid, mem_eval_docids_lst+upsample_candidate_docids_lst)  # [1, 768]; [mem_eval_size+upsample_num, 768]
                # p在q方向上的投影向量
                q_reps_norm = q_reps.norm(p=2, dim=-1, keepdim=True)  # [1, 1]
                p_q = torch.matmul(p_reps, q_reps.transpose(0, 1)) * q_reps / (q_reps_norm * q_reps_norm).clamp(min=1e-8)  # p在q方向上的投影向量, [mem_eval_size+upsample_num, 768]
                # p在q垂直方向上的向量
                p_q_vertical = p_reps - p_q   # [mem_eval_size+upsample_num, 768]
                mem_p_q_vertical = p_q_vertical[:mem_eval_size,:]  # [mem_eval_size, 768]
                candidate_p_q_vertical = p_q_vertical[mem_eval_size:,:]  # [upsample_num, 768]

                mem_p_q_vertical = mem_p_q_vertical.unsqueeze(dim=0).repeat(upsample_num,1,1)  # [upsample_num, mem_eval_size, 768]
                candidate_p_q_vertical = candidate_p_q_vertical.unsqueeze(dim=1).repeat(1,mem_eval_size,1)  # [upsample_num, mem_eval_size, 768]
                inter_dis = F.pairwise_distance(candidate_p_q_vertical.reshape(-1, candidate_p_q_vertical.size(-1)), mem_p_q_vertical.reshape(-1, mem_p_q_vertical.size(-1)), p=2.0)  # [upsample_num*mem_eval_size]
                inter_dis = inter_dis.reshape(candidate_p_q_vertical.size(0), candidate_p_q_vertical.size(1))
                inter_dis = torch.sum(inter_dis, dim=-1) / inter_dis.size(-1)  # [upsample_num], 尽可能大

                num_new = min(place_left, upsample_num)
                indices = inter_dis.sort(dim=0, descending=True)[1][:num_new]
                buffer.buffer_qid2dids[qid].extend(np.array(upsample_candidate_docids_lst)[indices.cpu()])

    def get_model_out(self, buffer, model_temp, qid, docids_lst):
        q_lst = [buffer.qid2query[qid]]
        doc_lst = [buffer.did2doc[did] for did in docids_lst]
        q_lst = buffer.tokenizer.batch_encode_plus(q_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        doc_lst = buffer.tokenizer.batch_encode_plus(doc_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        for key, value in q_lst.items():
            q_lst[key] = value.to(self.train_params.device)
        for key, value in doc_lst.items():
            doc_lst[key] = value.to(self.train_params.device)
        model_out = model_temp(q_lst, doc_lst)
        q_reps = model_out.q_reps
        p_reps = model_out.p_reps
        return q_reps, p_reps
