import torch
import torch.nn.functional as F
import collections
import copy
import numpy as np

from .buffer_utils import get_grad_vector, cosine_similarity


class GSSGreedyUpdate(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.mem_strength = params.gss_mem_strength  # 从memory中采样几个batch
        self.gss_batch_size = params.gss_batch_size   # 从memory中采样时每个batch的大小
        if kwargs['buffer_score']:
            print('load buffer score...')
            self.buffer_score = kwargs['buffer_score']  # 存储每个样本附带的score
        else:
            print('init buffer score...')
            self.buffer_score = collections.defaultdict(list)  # 存储每个样本附带的score
        self.params = params
        self.train_params = train_params

    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,1:] # 去掉pos passage
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,0]  # pos passage id

        model_temp = copy.deepcopy(buffer.model)
        grad_dims = []
        for param in model_temp.parameters():
            grad_dims.append(param.data.numel())
        
        q_lst, d_lst = kwargs['q_lst'], kwargs['d_lst']
        for i, qid in enumerate(qid_lst):
            cur_q_lst = {}  # [1, 32], cuda
            for key, val in q_lst.items():
                cur_q_lst[key] = val[i:i+1]
            cur_d_lst = {}  # [n, 128], cuda
            for key, val in d_lst.items():
                cur_d_lst[key] = val.reshape(batch_size, n_doc, -1)[i]
            docids = docids_neg_lst[i]  # 该query新来的 negative doc, array
            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left <= 0:  # 如果此时buffer已满
                batch_sim, mem_grads = self.get_batch_sim(buffer, model_temp, grad_dims, qid, docids_pos_lst[i], cur_q_lst, cur_d_lst)  # 从buffer随机采样几个batch计算他们的梯度，并计算batch_x整体与他们的梯度最大相似度， batch_sim=实数，mem_grads=[mem_strength,模型总参数量]
                if batch_sim < 0:
                    buffer_score = torch.Tensor(self.buffer_score[qid]).to(self.train_params.device)  # tensor
                    buffer_sim = (buffer_score - torch.min(buffer_score)) / ((torch.max(buffer_score) - torch.min(buffer_score)) + 0.01)
                    index = torch.multinomial(buffer_sim, len(docids), replacement=False)  # 按照标准化后的score采样出len(docids)个下标, tensor

                    batch_item_sim = self.get_each_batch_sample_sim(model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst)  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)  # 标准化到[0,1]
                    buffer_repl_batch_sim = ((buffer_score[index] + 1) / 2).unsqueeze(1)
                    outcome = torch.multinomial(torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1), 1, replacement=False)
                    added_indx = torch.arange(end=batch_item_sim.size(0))
                    sub_index = outcome.squeeze(1).bool()  # 由outcome决定是否替换

                    # 执行替换
                    buffer.buffer_qid2dids[qid] = np.array(buffer.buffer_qid2dids[qid])
                    self.buffer_score[qid] = np.array(self.buffer_score[qid])
                    buffer.buffer_qid2dids[qid][index[sub_index].cpu()] = docids[added_indx[sub_index].cpu()].copy()
                    self.buffer_score[qid][index[sub_index].cpu()] = batch_item_sim[added_indx[sub_index].cpu()].clone().cpu().numpy()
                    buffer.buffer_qid2dids[qid] = buffer.buffer_qid2dids[qid].tolist()
                    self.buffer_score[qid] = self.buffer_score[qid].tolist()
            else:   # 如果此时buffer未满，优先拿x中靠前的数据填满buffer，剩余的丢弃
                offset = min(place_left, len(docids))
                docids = docids[:offset]  # array
                if len(buffer.buffer_qid2dids[qid]) == 0:
                    batch_sample_memory_cos = torch.zeros(len(docids)) + 0.1   # 初始化score
                else:
                    mem_grads = self.get_rand_mem_grads(buffer, model_temp, grad_dims, qid, docids_pos_lst[i], cur_q_lst)  # 从buffer随机采样几个batch计算他们的梯度， mem_grads=[mem_strength,模型总参数量]
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst)  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                buffer.buffer_qid2dids[qid].extend(docids.tolist())
                self.buffer_score[qid].extend(batch_sample_memory_cos.tolist())

    def get_batch_sim(self, buffer, model_temp, grad_dims, qid, did_pos, q_lst, d_lst):
        mem_grads = self.get_rand_mem_grads(buffer, model_temp, grad_dims, qid, did_pos, q_lst)

        model_temp.zero_grad()
        loss = model_temp.forward(q_lst, d_lst, False).loss  # 不使用inbatch_loss
        loss.backward()
        batch_grad = get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device).unsqueeze(0)
        batch_sim = max(cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, buffer, model_temp, grad_dims, qid, did_pos, q_lst):
        # 从memory中随机采mem_strength个batch，batch_size大小为gss_batch_size，计算他们的梯度

        buffer_docid_lst = buffer.buffer_qid2dids[qid]  # list
        gss_batch_size = min(self.gss_batch_size, len(buffer_docid_lst))  # batch size
        num_mem_subs = min(self.mem_strength, len(buffer_docid_lst) // gss_batch_size)  # 采几个batch
        mem_grads = torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32).to(self.train_params.device)  # [num_mem_subs, grad_total_dims]
        shuffeled_inds = torch.randperm(len(buffer_docid_lst))

        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[i*gss_batch_size:i*gss_batch_size+gss_batch_size]
            docids_lst = np.array(buffer_docid_lst)[random_batch_inds]
            docids_lst = np.insert(docids_lst, 0, did_pos)

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
                doc_lst[key] = value.to(self.train_params.device)  # [gss_batch_size+1, 128]
            
            model_temp.zero_grad()
            loss = model_temp.forward(q_lst, doc_lst, False).loss  # 不使用inbatch_loss
            loss.backward()
            mem_grads[i].data.copy_(get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device))
        return mem_grads

    def get_each_batch_sample_sim(self, model_temp, grad_dims, mem_grads, q_lst, d_lst):
        # mem_grads是从memory中采样几个batch计算的几个梯度
        # 该函数用于计算batch_docids中每个新数据与mem_grads的cos最大值

        num_doc = d_lst['input_ids'].size(0)
        cosine_sim = torch.zeros(num_doc-1).to(self.train_params.device)
        for i in range(1, num_doc):
            doc_lst = {}
            for key, value in d_lst.items():
                doc_lst[key] = torch.cat((value[:1], value[i:i+1]), dim=0)  # [2, 128]

            model_temp.zero_grad()
            loss = model_temp.forward(q_lst, doc_lst, False).loss  # 不使用inbatch_loss
            loss.backward()
            this_grad = get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device).unsqueeze(0)
            cosine_sim[i-1] = max(cosine_similarity(mem_grads, this_grad))
        return cosine_sim
