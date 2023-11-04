import torch
import numpy as np
from collections import defaultdict
from collections import Counter
import random


def random_retrieve(buffer_dids, num_retrieve, excl_indices=None, return_indices=False):
    # buffer_dids = buffer.buffer_qid2dids[qid]
    # num_retrieve = 要返回多少个docid

    filled_indices = np.arange(len(buffer_dids))
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))  # 在filled_indices中不在excl_indices中的元素

    indices = list(np.random.choice(valid_indices, num_retrieve, replace=False))
    dids = list(np.array(buffer_dids)[indices])

    if return_indices:
        return dids, indices  # list, list
    else:
        return dids  # list


def get_grad_vector(pp, grad_dims, device):
    grads = torch.Tensor(sum(grad_dims)).to(device)
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim = torch.mm(x1, x2.t())/(w1 * w2.t()).clamp(min=eps)
    return sim


def cosine_similarity_3d(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=-1, keepdim=True)
    sim = torch.matmul(x1, x2.transpose(1, 2))/torch.matmul(w1, w2.transpose(1, 2)).clamp(min=eps)
    return sim
