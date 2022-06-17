from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
)
from .modeling_outputs import ContrastiveLearningOutput, SWAMOutput
from .SWAM import SWAM, SimpleHead
from .model_utils import CosSimilarityWithTemp, Pooler
class SWAMRobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kwargs.pop("model_args", None)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.swam = SWAM(config)
        self.similarity = CosSimilarityWithTemp(self.model_args.temp)
        self.simple_head = SimpleHead(config)
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_0: Optional[torch.LongTensor] = None,
        attention_mask_0: Optional[torch.FloatTensor] = None,
        input_ids_1: Optional[torch.LongTensor] = None,
        attention_mask_1: Optional[torch.FloatTensor] = None,
        input_ids_2: Optional[torch.LongTensor] = None,
        attention_mask_2: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        return_weights: Optional[bool] = None,
        inference: Optional[bool] = False,
        
    ) -> Union[Tuple[torch.Tensor], ContrastiveLearningOutput]:
        r"""
        test
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        potential_input_list = [input_ids, input_ids_0, input_ids_1, input_ids_2]
        potential_attention_list = [attention_mask, attention_mask_0, attention_mask_1, attention_mask_2]
        input_ids = [x for x in potential_input_list if x is not None]
        attention_mask = [x for x in potential_attention_list if x is not None]
        num_input_ids = len(input_ids)
        batch_size = input_ids[0].size(0)
        input_ids = input_ids * 2 if num_input_ids == 1 and not inference else input_ids
        input_ids = torch.concat(input_ids, dim=0)
        attention_mask = attention_mask * 2 if num_input_ids == 1 and not inference else attention_mask
        attention_mask = torch.concat(attention_mask, dim=0)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        word_embeddings = hidden_states[0]
        last_hidden_state = outputs.last_hidden_state
        swam_outputs, weights = self.swam(word_embeddings, last_hidden_state, attention_mask)
        swam_outputs = self.simple_head(swam_outputs) if not self.model_args.mlp_only_train or not inference else swam_outputs
        anchor_output = swam_outputs[:batch_size]
        positive_and_negative_output = swam_outputs[batch_size:]
        cos_sim = self.similarity(anchor_output.unsqueeze(1), positive_and_negative_output.unsqueeze(0)) if not inference else None
        labels = torch.arange(batch_size, dtype=torch.long, device=anchor_output.device) if not inference else None

        loss = CrossEntropyLoss()(cos_sim, labels) if not inference else None

        if not return_dict:
            raise ValueError("please implement the following block")
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SWAMOutput(
            loss=loss,
            hidden_states=anchor_output if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            swam_weights=weights if return_weights else None,
        )
class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kwargs.pop("model_args", None)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Initialize weights and apply final processing
        self.similarity = CosSimilarityWithTemp(self.model_args.temp)        
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)        
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_0: Optional[torch.LongTensor] = None,
        attention_mask_0: Optional[torch.FloatTensor] = None,
        input_ids_1: Optional[torch.LongTensor] = None,
        attention_mask_1: Optional[torch.FloatTensor] = None,
        input_ids_2: Optional[torch.LongTensor] = None,
        attention_mask_2: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        inference: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], ContrastiveLearningOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        potential_input_list = [input_ids, input_ids_0, input_ids_1, input_ids_2]
        potential_attention_list = [attention_mask, attention_mask_0, attention_mask_1, attention_mask_2]
        input_ids = [x for x in potential_input_list if x is not None]
        attention_mask = [x for x in potential_attention_list if x is not None]
        num_input_ids = len(input_ids)
        batch_size = input_ids[0].size(0)
        input_ids = input_ids * 2 if num_input_ids == 1 and not inference else input_ids
        input_ids = torch.concat(input_ids, dim=0)
        attention_mask = attention_mask * 2 if num_input_ids == 1 and not inference else attention_mask
        attention_mask = torch.concat(attention_mask, dim=0)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output  = self.pooler(attention_mask, outputs)
        anchor_output = pooler_output[:batch_size]
        positive_and_negative_output = pooler_output[batch_size:]
        cos_sim = self.similarity(anchor_output.unsqueeze(1), positive_and_negative_output.unsqueeze(0)) if not inference else None
        labels = torch.arange(batch_size, dtype=torch.long, device=anchor_output.device) if not inference else None

        loss = CrossEntropyLoss()(cos_sim, labels) if not inference else None

        if not return_dict:
            raise ValueError("please implement the following block")
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ContrastiveLearningOutput(
            loss=loss,
            hidden_states=anchor_output if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )