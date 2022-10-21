#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Converted SimCSE to Huggingface code style. Implemented Gumbel-Softmax based discrete optimization."""
import logging
import os
import random
import sys
from huggingface_hub import update_repo_visibility
import datasets
from datasets import load_dataset, load_metric
import importlib
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trainers.CustomTrainingArgument import CustomTrainingArgument
from arguments.default import DataTrainingArguments, ModelArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArgument))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task="sts",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    proxy_config = AutoConfig.from_pretrained(
        model_args.proxy_config if model_args.proxy_config_name else model_args.proxy_model_name_or_path,
        finetuning_task="sts",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_class = getattr(
        importlib.import_module(f"..{model_args.model_package_name}", package="models.subpkg"), 
        model_args.model_class_name
        )
    model_init_kwargs = {}
    for init_args in model_args.model_init_kwargs.split(";"):
        model_init_kwargs[init_args] = eval(init_args)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **model_init_kwargs,
    )
    

    # Preprocessing the raw_datasets
    # Padding strategy
    
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result_dict = {}
        for i, column in enumerate(column_names):
        
            result = tokenizer(examples[column], padding=padding, max_length=max_seq_length, truncation=True)
            for k, v in result.items():
                result_dict[k+"_"+str(i)] = v

        return result_dict
    if training_args.do_train:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # Initialize our Trainer
    trainer_class = getattr(
        importlib.import_module(f"..{model_args.trainer_package_name}", package="trainers.subpkg"),
        model_args.trainer_class_name,
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # handle temperature scheduler
    # train_dataloader = trainer.get_train_dataloader()
    # args=training_args
    # total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

    # len_dataloader = None
    # len_dataloader = len(train_dataloader)
    # num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    # num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    # num_examples = len(train_dataloader.dataset)
    # max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    # num_train_epochs = math.ceil(args.num_train_epochs)
    # num_train_samples = len(train_dataloader.dataset) * args.num_train_epochs

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_senteval_transfer=True)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.evaluate(eval_senteval_transfer=model_args.eval_transfer, metric_key_prefix="test", predict=True)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "sentence-similarity"}
    # if data_args.task_name is not None:
    kwargs["language"] = "en"
    # kwargs["dataset_tags"] = "glue"
    # kwargs["dataset_args"] = data_args.task_name
    # kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
        update_repo_visibility(training_args.hub_model_id, private=model_args.private)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()