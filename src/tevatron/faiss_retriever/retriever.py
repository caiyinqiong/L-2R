import numpy as np
import faiss
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)
    
    def convert_index_to_gpu(self, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(512*1024*1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            self.index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, self.index, co)
        else:
            gpu_resources = []
            if len(gpu_resources) == 0:
                import torch
                for i in range(torch.cuda.device_count()):
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(256*1024*1024*1024*1024)
                    gpu_resources.append(res)

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), total=num_query // batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
