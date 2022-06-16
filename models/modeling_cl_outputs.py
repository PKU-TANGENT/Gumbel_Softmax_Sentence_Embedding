from dataclasses import dataclass
from optparse import Option
import torch
from typing import List, Optional, Tuple, Union
from transformers.utils.generic import ModelOutput
@dataclass
class ContrastiveLearningOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SWAMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    swam_weights: Optional[Tuple[torch.FloatTensor]] = None