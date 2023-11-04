import torch
import torch.nn.functional as F
from .buffer_utils import random_retrieve, get_grad_vector
import copy
import numpy as np


class MIR_retrieve(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params
        self.subsample = params.subsample  # 先取一个big batch，再从中选择出eps_mem_batch个缓存数据返回
        self.num_retrieve = params.mem_batch_size  # memory data要使用的 batch size

    def retrieve(self, buffer, qid_lst, docids_lst, **kwargs):
        subsample = self.subsample
        for qid in qid_lst:
            subsample = min(subsample, len(buffer.buffer_qid2dids[qid]))
        if subsample == 0:
            return None

        # # 得到更新后的模型
        # grad_dims = []
        # for param in buffer.model.parameters():
        #     grad_dims.append(param.data.numel())  # numel() 获取元素个数
        # grad_vector = get_grad_vector(buffer.model.parameters, grad_dims, self.train_params.device)  # 有问题，因为此时的buffer.model没有梯度
        # model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims, kwargs['lr'])

        # 得到更新后的模型
        q_lst, d_lst = kwargs['q_lst'], kwargs['d_lst']
        model_temp = copy.deepcopy(buffer.model)
        model_temp.train()
        model_temp.zero_grad()
        loss = model_temp(q_lst, d_lst).loss
        loss.backward()
        with torch.no_grad():
            for param in model_temp.parameters():
                if param.grad is not None:
                    param.data = param.data - kwargs['lr'] * param.grad.data

        # retrieve
        docids_lst_from_new = np.array(docids_lst).reshape(len(qid_lst), -1)[:,:1]  # pos doc id, [num_q, 1]
        docids_lst_from_mem = []   # [num_q * (1+ subsample)]
        for i, qid in enumerate(qid_lst):
            sub_docids = random_retrieve(buffer.buffer_qid2dids[qid], subsample)
            docids_lst_from_mem.extend(docids_lst_from_new[i].tolist() + sub_docids)

        doc_lst = [buffer.did2doc[did] for did in docids_lst_from_mem]
        doc_lst = buffer.tokenizer.batch_encode_plus(doc_lst,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.params.p_max_len,
                    truncation='only_first',
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt")  # [num_q * (1+ subsample)]
        for key, value in doc_lst.items():
            doc_lst[key] = value.to(self.train_params.device)
        
        if self.params.compatible:
            identity = []
            doc_emb_from_mem = []
            for i, docid in enumerate(docids_lst_from_mem):
                docid = int(docid)
                if docid in buffer.buffer_did2emb:
                    identity.append(i)
                    doc_emb_from_mem.append(buffer.buffer_did2emb[docid])
            identity = torch.tensor(identity)
            doc_emb_from_mem = torch.tensor(np.array(doc_emb_from_mem), device=self.train_params.device)  # [num_q * (1+ subsample), 768]

        buffer.model.eval()
        model_temp.eval()
        with torch.no_grad():
            if self.params.compatible:
                res_pre = buffer.model.forward(q_lst, doc_lst, identity, doc_emb_from_mem)   # 更新前的模型, 返回的emb中已经替换成old emb
                res_post = model_temp.forward(q_lst, doc_lst, identity, doc_emb_from_mem)   # 更新后的模型, 返回的emb中已经替换成old emb
            else:
                res_pre = buffer.model.forward(q_lst, doc_lst)   # 更新前的模型
                res_post = model_temp.forward(q_lst, doc_lst)   # 更新后的模型
        buffer.model.train()

        loss_pre = self.cal_loss(res_pre)  # [num_q, subsample]
        loss_post = self.cal_loss(res_post)  # [num_q, subsample]

        loss = loss_post - loss_pre   # [num_q, subsample]
        num_retrieve = min(self.num_retrieve, subsample)
        indexs = loss.sort(dim=-1, descending=True)[1][:,:num_retrieve]  # [num_q, num_retrieve]

        doc_lst_from_mem = {}
        for key, val in doc_lst.items():
            doc_lst_from_mem[key] = torch.gather(
                val.reshape(len(qid_lst), -1, val.size(-1))[:,1:,:],
                1,
                indexs.unsqueeze(dim=-1).repeat(1, 1, val.size(-1)))
        if self.params.compatible:
            docids_lst_from_mem = torch.tensor(docids_lst_from_mem)
            docids_lst_from_mem = torch.gather(
                docids_lst_from_mem.reshape(len(qid_lst), -1)[:,1:],
                1,
                indexs.to('cpu'))
            return docids_lst_from_mem, doc_lst_from_mem  # [num_q, mem_batch_size], cpu; [num_q, mem_batch_size, doc_len], gpu
        return doc_lst_from_mem  # [num_q, mem_batch_size, doc_len], gpu

    def get_future_step_parameters(self, model, grad_vector, grad_dims, lr):
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)  # 用grad_vector设置new_model.parameters的梯度
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - lr * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def cal_loss(self, model_output):
        q_reps = model_output.q_reps  # [num_q, 768]
        p_reps = model_output.p_reps  # [num_q * n, 768]

        num_q = q_reps.size(0)
        n_psg = p_reps.size(0) // q_reps.size(0)

        p_reps = p_reps.reshape(num_q, n_psg, -1)  # [num_q, n_psg, dim]
        scores = torch.matmul(q_reps.unsqueeze(dim=1), p_reps.transpose(1,2)).squeeze(dim=1)  # [num_q, n_psg]

        scores_pos = scores[:,:1].repeat(1, n_psg-1).reshape(-1)  # [num_q*(n_psg-1)]
        scores_neg = scores[:,1:].reshape(-1)  # [num_q*(n_psg-1)]
        scores = torch.stack((scores_pos, scores_neg), dim=1)  # [num_q*(n_psg-1), 2]
        target = torch.tensor([0]*scores.size(0), device=scores.device, dtype=torch.long)  # [num_q*(n_psg-1)]
        loss = F.cross_entropy(scores, target, reduction='none').reshape(num_q, n_psg-1)  # [num_q, n_psg-1]
        return loss
