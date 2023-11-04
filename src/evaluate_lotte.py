import argparse
from collections import defaultdict
import json
import os


def evaluate_dataset(k, query_path, data_path, rankings_path):
    eval_query = set()
    with open(query_path, mode="r") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["qid"])
            eval_query.add(qid)

    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid, _, pid, rank, _, _ = items
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            if qid in eval_query:
                rankings[qid].append(pid)
                assert rank == len(rankings[qid])

    success = 0
    num_q = 0
    recall = 0.0

    with open(data_path, mode="r") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["qid"])
            answer_pids = set(data["answer_pids"])
            if qid in eval_query:
                num_q += 1
                hit = set(rankings[qid][:k]).intersection(answer_pids)
                if len(hit) > 0:
                    success += 1
                    recall += (len(hit)/len(answer_pids))
    print(
        f"# query: {len(rankings)}: {num_q}\n",
        f"Success@{k}: {success / len(rankings) * 100:.1f}\n"
        f"Recall@{k}: {recall / len(rankings) * 100:.1f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTTE evaluation script")
    parser.add_argument("--k", type=int, default=5, help="Success@k")
    parser.add_argument(
        "-q", "--query_path", type=str, required=True, help="Path to LoTTE data directory"
    )
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Path to LoTTE data directory"
    )
    parser.add_argument(
        "-r",
        "--rankings_path",
        type=str,
        required=True,
        help="Path to LoTTE rankings directory",
    )
    args = parser.parse_args()
    evaluate_dataset(args.k, args.query_path, args.data_path, args.rankings_path)
