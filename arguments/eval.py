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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    pretrained_path: str = field(
        metadata={"help": "Path or model identified to jointly trained weights by this repo."}
    )
    model_name_or_path: str = field(
        metadata={"help": "For backbone model, path to model or model identifier from huggingface.co/models"}
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
    trainer_class_name: str = field(
        default="STSTrainer",
        metadata={"help": "Name of the trainer class to use."},
    )
    trainer_package_name: str = field(
        default="STSTrainer",
        metadata={"help": "Name of the trainer package to use."},
    )
    model_init_kwargs: str = field(
        default="model_args",
        metadata={"help": "Name of arguments to pass to `__init__` function, separated by ;"},
    )