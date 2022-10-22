#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
model_name_or_path="checkpoint/gumbel_softmax/unsup-bert-base-uncased-avg/checkpoint-46875"
# proxy_model must have the same tokenizer system as the base model
if [[ "$model_name_or_path" =~ "roberta" ]];then
    model_architecture=roberta
    proxy_model=distilroberta-base
else
    model_architecture=bert
    proxy_model=distilbert-base-uncased
fi
pooler_type="avg"
output_dir="checkpoint/eval_gumbel_softmax/${model_name_or_path//\//-}"
export WANDB_DISABLED=true
export WANDB_PROJECT=$model_name_or_path
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client eval.py \
python eval.py \
    --model_name_or_path $model_name_or_path \
    --proxy_model_name_or_path $proxy_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --learning_rate 5e-5 \
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
    --model_class_name "GumbelSoftmaxPLMForCL" \
    --model_package_name "modeling_gumbel_softmax_cl" \
    --ignore_transfer_test \
    --model_init_kwargs "model_args;config;proxy_config" \
    --overwrite_output_dir \
    --compute_sparsity True \
    # --hub_model_id $hub_model_id \
    # --push_to_hub \
