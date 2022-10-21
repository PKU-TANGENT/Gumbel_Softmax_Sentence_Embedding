#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
contrastive_learning_style="unsup"
model_name_or_path="roberta-base"
# export CUDA_VISIBLE_DEVICES=$1
# contrastive_learning_style=$2
# model_name_or_path=$3
if [[ "${contrastive_learning_style}" == "unsup" ]]; then
    train_file="JeremiahZ/simcse_unsup_wiki"
else
    train_file="JeremiahZ/simcse_sup_nli"
fi
if [[ "$model_name_or_path" =~ "roberta" ]];then
    model_architecture=roberta
else
    model_architecture=bert
fi
if [[ "${contrastive_learning_style}" == "unsup" ]]; then
    if [[ "${model_architecture}" == "roberta" ]]; then
        learning_rate=1e-5
        per_device_train_batch_size=512
    else
        learning_rate=3e-5
        per_device_train_batch_size=64
    fi
else
    per_device_train_batch_size=512
    learning_rate=5e-5
fi

export TOKENIZERS_PARALLELISM=false
pooler_type="avg"
output_dir="checkpoint/reproduce/${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
hub_model_id="reproduce-${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
export WANDB_PROJECT=$model_name_or_path
export WANDB_DISABLED=true
# python train.py \
python -m debugpy --listen 127.0.0.1:9999 --wait-for-client train.py \
    --model_name_or_path $model_name_or_path \
    --train_file $train_file \
    --num_train_epochs 3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --learning_rate $learning_rate \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --metric_for_best_model stsbenchmark_train_and_dev_spearman \
    --load_best_model_at_end \
    --pooler_type $pooler_type \
    --temp 0.05 \
    --do_eval \
    --do_predict \
    --fp16 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --load_best_model_at_end \
    --greater_is_better True \
    --private \
    --do_train \
    --hub_model_id $hub_model_id \
    --model_class_name "${model_architecture^}ForCL" \
    --model_package_name "modeling_${model_architecture}_cl" \
    --ignore_transfer_test \
    --model_head_lr $learning_rate \
    # --push_to_hub \
    # --overwrite_output_dir \
