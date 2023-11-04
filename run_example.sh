
# #################################################################################
# #################################################################################
# #################################################################################
CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.train \
  --output_dir model_lotte_s2 \
  --model_name_or_path ../zaug_er_s1_top500_sample100_ep15/model_lotte \
  --save_steps 2000 \
  --train_dir ../data_incre_aug/session_2/train_dir_top200_sample100 \
  --data_cache_dir ../data_incre_aug/session_2/lotte-train2_t200_s100-cache \
  --fp16 \
  --dataloader_num_workers 3 \
  --per_device_train_batch_size 16 \
  --train_n_passages 8 \
  --learning_rate 1e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 12 \
  --logging_steps 50 \
  --buffer_data ../zaug_er_s1_top500_sample100_ep15/model_lotte \
  --query_data ../data_incre_aug/train_dir/train.jsonl \
  --doc_data ../data_incre_aug/session_all2/corpus_dir/all_corpus.jsonl \
  --mem_size 30 \
  --mem_batch_size 2 \
  --cl_method 'our' \
  --retrieve_method 'our_emb_cosine' \
  --update_method 'our_emb_cosine' \
  --new_batch_size 2 \
  --alpha 0.6 \
  --beta 0.4 \
  --mem_upsample 6 \
  --mem_eval_size 10 \
  --mem_replace_size 10 \
  --upsample_scale 2.0 \
  --compatible \
  --compatible_ce_alpha 1.0 \


# #################################################################################
# #################################################################################
# #################################################################################
ckpt=""     # 修改model路径

mkdir -p emb_lotte_ckpt${ckpt}_2
for s in $(seq -f "%02g" 0 19)
do
CUDA_VISIBLE_DEVICES=3 python -m tevatron.driver.encode \
  --output_dir emb_lotte_ckpt${ckpt}_2 \
  --model_name_or_path model_lotte_s2 \
  --data_cache_dir ../data_incre_aug/session_2/lotte-corpus2-cache \
  --fp16 \
  --dataloader_num_workers 2 \
  --per_device_eval_batch_size 512 \
  --p_max_len 256 \
  --encode_in_path ../data_incre_aug/session_2/corpus_dir/all_corpus.jsonl \
  --encoded_save_path emb_lotte_ckpt${ckpt}_2/corpus_emb.${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s}
done

mkdir -p emb_lotte_ckpt${ckpt}_2
CUDA_VISIBLE_DEVICES=3 python -m tevatron.driver.encode \
  --output_dir emb_lotte_ckpt${ckpt}_2 \
  --model_name_or_path model_lotte_s2 \
  --data_cache_dir ../data_incre_aug/session_2/lotte-test2-cache \
  --fp16 \
  --dataloader_num_workers 2 \
  --per_device_eval_batch_size 512 \
  --encode_in_path ../data_incre_aug/session_2/test_dir/test.jsonl \
  --encoded_save_path emb_lotte_ckpt${ckpt}_2/query_test_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry

mkdir -p retrieval_lotte_ckpt${ckpt}_2
CUDA_VISIBLE_DEVICES=3 python -m tevatron.faiss_retriever \
--query_reps emb_lotte_ckpt${ckpt}_2/query_test_emb.pkl \
--passage_reps "../zaug_er_s1_top500_sample100_ep15/emb_lotte_ckpt_1/corpus_emb.*.pkl" "emb_lotte_ckpt${ckpt}_2/corpus_emb.*.pkl" \
--depth 1000 \
--batch_size 1024 \
--save_text \
--save_ranking_to retrieval_lotte_ckpt${ckpt}_2/rank.test.txt \

python -m tevatron.utils.format.convert_result_to_trec \
              --input retrieval_lotte_ckpt${ckpt}_2/rank.test.txt \
              --output retrieval_lotte_ckpt${ckpt}_2/rank.test.txt.trec

python ../src/evaluate_lotte.py --k 5 -q ../data_incre_aug/session_2/test_qrel.jsonl -d ../data_incre_aug/session_all2/test_qrel.jsonl -r retrieval_lotte_ckpt${ckpt}_2/rank.test.txt.trec
python ../src/evaluate_lotte.py --k 20 -q ../data_incre_aug/session_2/test_qrel.jsonl -d ../data_incre_aug/session_all2/test_qrel.jsonl -r retrieval_lotte_ckpt${ckpt}_2/rank.test.txt.trec
python ../src/evaluate_lotte.py --k 100 -q ../data_incre_aug/session_2/test_qrel.jsonl -d ../data_incre_aug/session_all2/test_qrel.jsonl -r retrieval_lotte_ckpt${ckpt}_2/rank.test.txt.trec
python ../src/evaluate_lotte.py --k 1000 -q ../data_incre_aug/session_2/test_qrel.jsonl -d ../data_incre_aug/session_all2/test_qrel.jsonl -r retrieval_lotte_ckpt${ckpt}_2/rank.test.txt.trec

# rm -rf emb_lotte_ckpt${ckpt}_2

python -m tevatron.buffer.generate_buffer_emb \
--buffer_file ./model_lotte_s2/buffer.pkl \
--train_file ../data_incre_aug/train_dir/train.jsonl \
--passage_reps "./emb_lotte_ckpt${ckpt}_2/corpus_emb.*.pkl" \
--old_buffer_emb_file ../zaug_er_s1_top500_sample100_ep15/model_lotte/buffer_emb.pkl \
--new_buffer_emb_file ./model_lotte_s2
