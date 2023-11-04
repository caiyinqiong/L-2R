import torch
import torch.nn.functional as F
import collections
import copy
import numpy as np

from .buffer_utils import get_grad_vector, cosine_similarity, random_retrieve


class OurUpdate(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params

        self.mem_eval_size = params.mem_eval_size
        self.mem_replace_size = params.mem_replace_size
        self.upsample_scale = params.upsample_scale

        self.pos_docids = collections.defaultdict(set)
        self.candidate_neg_docids = collections.defaultdict(set)

    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        pos_docids = kwargs['pos_docids']
        for qid, docids in pos_docids.items():
            self.pos_docids[qid].update(docids)
        
        candidate_neg_docids = kwargs['candidate_neg_docids']
        for qid, docids in candidate_neg_docids.items():
            self.candidate_neg_docids[qid].update(docids)
    
    def replace(self, buffer):
        model_temp = copy.deepcopy(buffer.model)
        grad_dims = []
        for param in model_temp.parameters():
            grad_dims.append(param.data.numel())
        
        for qid, candidate_docids in self.candidate_neg_docids.items():
            candidate_docids = np.array(list(candidate_docids))

            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left == 0:  # buffer满， 执行替换
                mem_eval_size = min(self.mem_eval_size, len(buffer.buffer_qid2dids[qid]))
                mem_eval_docids, mem_eval_indices = random_retrieve(buffer.buffer_qid2dids[qid], mem_eval_size, return_indices=True)
                pos_docid = np.random.choice(list(self.pos_docids[qid]), 1).tolist()
                mem_eval_docids_lst = pos_docid + mem_eval_docids
                qid_lst = [qid]
                mem_eval_grad = self.get_batch_grad(buffer, model_temp, grad_dims, qid_lst, mem_eval_docids_lst)  # [参数个数]

                upsample_num = min(len(buffer.buffer_qid2dids[qid])-mem_eval_size, int(self.mem_replace_size*self.upsample_scale))  # 采样
                upsample_mem_docids, upsample_mem_indices = random_retrieve(buffer.buffer_qid2dids[qid], upsample_num, excl_indices=mem_eval_indices, return_indices=True)
                mem_docids_lst = pos_docid + upsample_mem_docids
                grad = self.get_each_grad(buffer, model_temp, grad_dims, qid_lst, mem_docids_lst)  # [upsample_num, 参数个数]
                mem_sim = cosine_similarity(grad, mem_eval_grad.unsqueeze(0)).squeeze(dim=-1)
                indices = mem_sim.sort(dim=0, descending=True)[1]  # 选相似度最大的一些

                upsample_num = min(len(candidate_docids), int(self.mem_replace_size*self.upsample_scale))  # 采样
                upsample_candidate_docids = random_retrieve(candidate_docids, upsample_num)  # list
                new_docids_lst = pos_docid + upsample_candidate_docids
                grad = self.get_each_grad(buffer, model_temp, grad_dims, qid_lst, new_docids_lst)  # [upsample_num，参数个数]
                new_sim = cosine_similarity(grad, mem_eval_grad.unsqueeze(0)).squeeze(dim=-1)  # [upsample_num]
                new_indices = new_sim.sort(dim=0, descending=False)[1]  # 选相似度最小的一些

                mem_replace_size = min([self.mem_replace_size, len(mem_sim), len(new_sim)])
                indices = indices[:mem_replace_size]
                new_indices = new_indices[:mem_replace_size]
                # 执行替换
                buffer.buffer_qid2dids[qid] = np.array(buffer.buffer_qid2dids[qid])
                buffer.buffer_qid2dids[qid][np.array(upsample_mem_indices)[indices.cpu()]] = np.array(upsample_candidate_docids)[new_indices.cpu()].copy()
                buffer.buffer_qid2dids[qid] = buffer.buffer_qid2dids[qid].tolist()
            
            elif place_left == buffer.buffer_size:  # buffer 全空，随机选择新数据放入
                all_indices = np.arange(len(candidate_docids))
                num_retrieve = min(len(candidate_docids), buffer.buffer_size)
                indices = list(np.random.choice(all_indices, num_retrieve, replace=False))
                buffer.buffer_qid2dids[qid].extend(candidate_docids[indices])

            else:  # buffer 有剩余，选择与eval mem最不接近的place_left个新数据，放入
                mem_eval_size = min(self.mem_eval_size, len(buffer.buffer_qid2dids[qid]))
                mem_eval_docids = random_retrieve(buffer.buffer_qid2dids[qid], mem_eval_size)
                pos_docid = np.random.choice(list(self.pos_docids[qid]), 1).tolist()
                mem_eval_docids_lst = pos_docid + mem_eval_docids
                qid_lst = [qid]
                mem_eval_grad = self.get_batch_grad(buffer, model_temp, grad_dims, qid_lst, mem_eval_docids_lst)

                upsample_num = min(len(candidate_docids), int(place_left*self.upsample_scale))  # 两倍量的采样
                upsample_candidate_docids = random_retrieve(candidate_docids, upsample_num)  # list
                new_docids_lst = pos_docid + upsample_candidate_docids
                candidate_grad = self.get_each_grad(buffer, model_temp, grad_dims, qid_lst, new_docids_lst)  # [upsample_num，参数个数]

                sim = cosine_similarity(candidate_grad, mem_eval_grad.unsqueeze(0)).squeeze(dim=-1)  # [candidate个数]
                num_new = min(place_left, len(sim))
                indices = sim.sort(dim=0, descending=False)[1][:num_new]  # 选相似度最小的一些
                buffer.buffer_qid2dids[qid].extend(np.array(upsample_candidate_docids)[indices.cpu()])

    def get_batch_grad(self, buffer, model_temp, grad_dims, qid_lst, docids_lst):
        q_lst = [buffer.qid2query[qid] for qid in qid_lst]
        q_lst = buffer.tokenizer.batch_encode_plus(q_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        for key, value in q_lst.items():
            q_lst[key] = value.to(self.train_params.device)

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

        model_temp.zero_grad()
        loss = model_temp.forward(q_lst, doc_lst, False).loss  # 不使用inbatch_loss
        loss.backward()
        mem_eval_grad = get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device)
        return mem_eval_grad
    
    def get_each_grad(self, buffer, model_temp, grad_dims, qid_lst, docids_lst):
        q_lst = [buffer.qid2query[qid] for qid in qid_lst]
        q_lst = buffer.tokenizer.batch_encode_plus(q_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        for key, value in q_lst.items():
            q_lst[key] = value.to(self.train_params.device)

        d_lst = [buffer.did2doc[did] for did in docids_lst]
        d_lst = buffer.tokenizer.batch_encode_plus(d_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
        for key, value in d_lst.items():
            d_lst[key] = value.to(self.train_params.device)
        
        num_doc = d_lst['input_ids'].size(0)
        mem_grads = torch.zeros(num_doc-1, sum(grad_dims), dtype=torch.float32).to(self.train_params.device)
        for i in range(1, num_doc):
            doc_lst = {}
            for key, value in d_lst.items():
                doc_lst[key] = torch.cat((value[:1], value[i:i+1]), dim=0)  # [2, 128]

            model_temp.zero_grad()
            loss = model_temp.forward(q_lst, doc_lst, False).loss  # 不使用inbatch_loss
            loss.backward()
            mem_grads[i-1].data.copy_(get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device))
        return mem_grads
