#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
contrastive_learning_style="unsup"
model_name_or_path="checkpoint/unsup-roberta-base"
# model_name_or_path="princeton-nlp/unsup-simcse-roberta-base"
# pooler_type="cls"
pooler_type="swam"
output_dir="checkpoint/${contrastive_learning_style}-${model_name_or_path}"
hub_model_id="${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
logging_steps=125

# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client train.py \
python train.py \
    --model_name_or_path $model_name_or_path \
    --output_dir $output_dir \
    --pooler_type $pooler_type \
    --temp 0.05 \
    --private \
    --do_predict \
    --train_file data/wiki1m_for_simcse.txt \
    --model_package_name modeling_roberta_cl \
    --mlp_only_train \
    # --model_class_name RobertaForCL \
    # --eval_transfer \
    # --num_train_epochs 1 \
    # --per_device_train_batch_size 64 \
    # --learning_rate 3e-5 \
    # --max_seq_length 32 \
    # --evaluation_strategy steps \
    # --save_strategy steps \
    # --eval_steps $logging_steps \
    # --save_steps $logging_steps \
    # --logging_steps $logging_steps \
    # --metric_for_best_model stsbenchmark_spearman \
    # --load_best_model_at_end \
    # --fp16 \
    # --gradient_accumulation_steps 1 \
    # --warmup_ratio 0.06 \
    # --weight_decay 0.1 \
    # --save_total_limit 1 \
    # --hub_model_id $hub_model_id \
    # --load_best_model_at_end \
    # --greater_is_better True \
    # --do_train \
    # --do_eval \
    # --push_to_hub \
