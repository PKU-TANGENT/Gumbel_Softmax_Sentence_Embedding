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
from .model_utils import CosSimilarityWithTemp
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
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], ContrastiveLearningOutput]:
        r"""
        test
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        name_input_ids = sorted([x for x in kwargs if x.startswith("input_ids")])
        input_ids = [kwargs.pop(x) for x in name_input_ids]
        name_attention_mask = sorted([x for x in kwargs if x.startswith("attention_mask")])
        attention_mask = [kwargs.pop(x) for x in name_attention_mask]
        num_input_ids = len(name_input_ids)
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
        swam_outputs = self.simple_head(swam_outputs)
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
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ContrastiveLearningOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ContrastiveLearningOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )