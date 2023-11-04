import logging
import os
import sys
from contextlib import nullcontext
from collections import defaultdict

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import RankDataset, RankCollator
from tevatron.modeling import EncoderOutput, ColbertModel
from tevatron.datasets import HFRankDataset

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = ColbertModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    rank_dataset = HFRankDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rank_dataset = RankDataset(data_args, rank_dataset.process(), tokenizer)

    rank_loader = DataLoader(
        rank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=RankCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    model = model.to(training_args.device)
    model.eval()
    qid_list = []
    did_list = []
    score_list = []
    for (q_ids, d_ids, query, doc) in tqdm(rank_loader):
        qid_list.extend(q_ids)
        did_list.extend(d_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in query.items():
                    query[k] = v.to(training_args.device)
                for k, v in doc.items():
                    doc[k] = v.to(training_args.device)
                model_output: EncoderOutput = model(query=query, passage=doc)
                score_list.extend(model_output.scores.cpu().detach().numpy().tolist())
    assert len(qid_list) == len(did_list) == len(score_list) 

    score_dict = defaultdict(list)
    for qid, did, score in zip(qid_list, did_list, score_list):
        score_dict[qid].append((float(score), did))

    with open(os.path.join(training_args.output_dir, 'rank.txt'), "w") as outFile:
        for query_id, para_lst in score_dict.items():
            para_lst = sorted(para_lst, key=lambda x:x[0], reverse=True)
            for score, para_id in para_lst:
                outFile.write("{}\t{}\t{}\n".format(query_id, para_id, score))


if __name__ == "__main__":
    main()
