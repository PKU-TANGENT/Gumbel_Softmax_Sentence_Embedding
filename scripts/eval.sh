#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
# contrastive_learning_style="unsup"
model_name_or_path="JeremiahZ/reproduce-sup-roberta-base-avg"
if [[ "$model_name_or_path" =~ "roberta" ]];then
    model_architecture=roberta
else
    model_architecture=bert
fi
pooler_type="avg"
output_dir="checkpoint/eval_reproduce/${model_name_or_path//\//-}"
# output_dir="checkpoint/gumbel_softmax/${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
# hub_model_id="gumbel_softmax-${contrastive_learning_style}-${model_name_or_path}-${pooler_type}"
export WANDB_DISABLED=true
export WANDB_PROJECT=$model_name_or_path
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client train.py \
python eval.py \
    --model_name_or_path $model_name_or_path \
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
    --model_class_name "${model_architecture^}ForCL" \
    --model_package_name "modeling_${model_architecture}_cl" \
    --ignore_transfer_test \
    --model_init_kwargs "model_args;config" \
    --overwrite_output_dir \
    # --hub_model_id $hub_model_id \
    # --push_to_hub \
