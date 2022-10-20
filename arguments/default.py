from dataclasses import dataclass, field
from typing import Optional
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A txt, csv or json file containing the training data."}
    )
    # SimCSE
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        elif self.train_file is None:
            raise ValueError("a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["txt", "csv", "json"], "`train_file` should be a txt, csv or a json file."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    proxy_model_name_or_path: str = field(
        metadata={"help": "For proxy model, path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    proxy_config_name: Optional[str] = field(
        default=None, metadata={"help": "For proxy model, pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name. Note that proxy model and backbone model must use the same tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    private: bool = field(
        default=False,
        metadata={"help": "Whether to create private repo on huggingface."},
    )
    
    # SimCSE Arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="avg",
        metadata={
            "help": 
            "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    eval_transfer: bool = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the model on the transfer task during training."
        }
    )
    ignore_transfer_test: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip transfer tasks at test."
        }    
    )
    # Arguments for Modularized SimCSE
    model_class_name: str = field(
        default="RobertaForCL",
        metadata={"help": "Name of the model class to use."},
    )
    model_package_name: str = field(
        default="modeling_roberta_cl",
        metadata={"help": "Name of the model package to use."},
    )
    model_head_lr: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for the model head."},
    )
    trainer_class_name: str = field(
        default="STSTrainer",
        metadata={"help": "Name of the trainer class to use."},
    )
    trainer_package_name: str = field(
        default="STSTrainer",
        metadata={"help": "Name of the trainer package to use."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the PLM backbone."},
    )
    model_pretrained_init: bool = field(
        default=False,
        metadata={"help": "Whether to init model from pretrained."},
    )
    model_init_kwargs: str = field(
        default="model_args",
        metadata={"help": "Name of arguments to pass to `__init__` function, separated by ;"},
    )
    init_tau: float = field(
        default=0.5,
        metadata={"help": "Init tau value for temparature scheduler."},
    )
    min_tau: float = field(
        default=0.5,
        metadata={"help": "Min tau value for temparature scheduler."},
    )
    exp_scheduler_hyper: float = field(
        default=1e-5,
        metadata={"help": "Hyper parameter `r` in exp scheduler `exp(-rt)`, where `t` is global training step."},
    )