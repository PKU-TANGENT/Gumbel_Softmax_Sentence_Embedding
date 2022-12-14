from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import (
    CrossEntropyLoss,
)
from .modeling_outputs import ContrastiveLearningOutput
from .model_utils import CosSimilarityWithTemp, Pooler, SimpleHead
class BertForCL(BertPreTrainedModel):
    def __init__(self, config, *inputs, **model_kwargs):
        super().__init__(config)
        model_args = model_kwargs.pop("model_args", None)
        self.model_args = model_args
        self.bert = BertModel(config, add_pooling_layer=False)
        config.pooler_type = model_args.pooler_type if model_args is not None else "cls"
        config.temp = model_args.temp if model_args is not None else 0.05
        self.config = config
        self.similarity = CosSimilarityWithTemp(config.temp)        
        self.pooler_type = config.pooler_type
        self.pooler = Pooler(config.pooler_type)   
        self.simple_head = SimpleHead(config)     
        self.loss_func = CrossEntropyLoss()      
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
        device=input_ids[0].device
        input_ids = torch.concat(input_ids, dim=0)
        attention_mask = attention_mask * 2 if num_input_ids == 1 and not inference else attention_mask
        attention_mask = torch.concat(attention_mask, dim=0)
        outputs = self.bert(
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
        if not inference or not self.model_args.mlp_only_train:
            pooler_output = self.simple_head(pooler_output)
        anchor_output = pooler_output[:batch_size]
        positive_and_negative_output = pooler_output[batch_size:]
        cos_sim = self.similarity(anchor_output.unsqueeze(1), positive_and_negative_output.unsqueeze(0)) if not inference else None
        labels = torch.arange(batch_size, dtype=torch.long, device=device) if not inference else None

        loss = self.loss_func(cos_sim, labels) if not inference else None

        if not return_dict:
            # raise ValueError("please implement the following block")
            output = (anchor_output,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ContrastiveLearningOutput(
            loss=loss,
            hidden_states=anchor_output if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )