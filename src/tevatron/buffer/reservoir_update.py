import torch
import numpy as np


class Reservoir_update(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()

    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,1:].tolist() # 去掉pos passage

        filled_idx_lst = []
        for i, qid in enumerate(qid_lst):
            docids = docids_lst[i]
            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left:
                offset = min(place_left, n_doc-1)
                buffer.buffer_qid2dids[qid].extend(docids[:offset])
                buffer.n_seen_so_far[qid] += offset
                if offset == n_doc -1:
                    filled_idx = list(range(len(buffer.buffer_qid2dids[qid])-offset, len(buffer.buffer_qid2dids[qid])))
                    filled_idx_lst.append(filled_idx)
                    continue

            docids = docids[place_left:]

            indices = torch.FloatTensor(len(docids)).uniform_(0, buffer.n_seen_so_far[qid]).long()  # 从[0, buffer.n_seen_so_far]等概率的采样出query.size(0)个数
            valid_indices = (indices < buffer.buffer_size).long()
            idx_new_data = valid_indices.nonzero().squeeze(-1)
            idx_buffer   = indices[idx_new_data]
            buffer.n_seen_so_far[qid] += len(docids)

            if idx_buffer.numel() == 0:
                filled_idx_lst.append([])
                continue

            assert idx_buffer.max() < buffer.buffer_size
            assert idx_new_data.max() < len(docids)

            idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(len(idx_buffer))}  # idx_buffer[i]：buffer的index，idx_new_data[i]：新数据的index
            data = np.array(buffer.buffer_qid2dids[qid])
            data[list(idx_map.keys())] = np.array(docids)[list(idx_map.values())]
            buffer.buffer_qid2dids[qid] = list(data)
            filled_idx_lst.append(list(idx_map.keys()))

        return filled_idx_lst  # list:[num_q, 每个q下被替换掉的buffer下标]
