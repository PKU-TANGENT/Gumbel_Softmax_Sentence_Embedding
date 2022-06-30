from transformers import TrainingArguments
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class CustomTrainingArgument(TrainingArguments):
    eval_ratio: Optional[float] = field(
        default=1,
        metadata={
            "help": "The ratio of a epoch to trigger eval."
        }
    )
    def __post_init__(self):
        self.gradient_accumulation_steps = 2 if self.per_device_train_batch_size >= 512 else 1
        self.per_device_train_batch_size = self.per_device_train_batch_size // self.gradient_accumulation_steps
        tmp_ratio = (self.per_device_train_batch_size * self.gradient_accumulation_steps) // 64
        self.logging_steps = 625 // (self.eval_ratio * tmp_ratio)
        self.eval_steps = self.logging_steps
        self.save_steps = self.logging_steps
        return super().__post_init__()


