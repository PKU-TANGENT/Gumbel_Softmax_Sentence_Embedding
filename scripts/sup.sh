#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
contrastive_learning_style="sup"
model_name_or_path="roberta-base"
pooler_type="swam"
output_dir="checkpoint/${contrastive_learning_style}-${model_name_or_path}"
hub_model_id="${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
logging_steps=125

# python -m debugpy --listen 127.0.0.1:5678 --wait-for-client train.py \
python train.py \
    --model_name_or_path $model_name_or_path \
    --train_file data/nli_for_simcse.csv \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps $logging_steps \
    --save_steps $logging_steps \
    --logging_steps $logging_steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --pooler_type $pooler_type \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --hub_model_id $hub_model_id \
    --push_to_hub \
    --load_best_model_at_end \
    --greater_is_better True \
    --private
