from transformers import (
    PreTrainedModel,
)
import math
from .modeling_proxy import ProxyBert, ProxyRoberta
from .modeling_gumbel_softmax_bert import GumbelSoftmaxBertModel
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from .modeling_outputs import ContrastiveLearningOutput 
from .modeling_outputs import ContrastiveLearningOutput 
from .model_utils import CosSimilarityWithTemp, Pooler, SimpleHead
# from .gumbel_softmax import get_gumbel_softmax_mask
class GumbelSoftmaxPLMForCL(PreTrainedModel):
    def __init__(self, config, proxy_config, *inputs, **model_kwargs):
        super().__init__(config)
        model_args = model_kwargs.pop("model_args", None)
        self.model_args = model_args
        config.pooler_type = model_args.pooler_type
        config.temp = model_args.temp
        self.config = config
        self.similarity = CosSimilarityWithTemp(config.temp)        
        self.pooler_type = config.pooler_type
        self.pooler = Pooler(config.pooler_type)   
        self.simple_head = SimpleHead(config)     
        self.loss_func = nn.CrossEntropyLoss()      
        # TODO: add roberta
        model_class, proxy_class = (GumbelSoftmaxBertModel, ProxyBert) if "roberta" not in config._name_or_path else (None, ProxyRoberta)
        self.model = model_class.from_pretrained(config._name_or_path, config=config)
        self.proxy = proxy_class.from_pretrained(proxy_config._name_or_path, config=proxy_config)

        # for temperature scheduler
        self.min_tau = model_args.min_tau
        self.exp_scheduler_hyper = model_args.exp_scheduler_hyper
        self.t = 0
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        potential_input_list = [input_ids, input_ids_0, input_ids_1, input_ids_2]
        potential_attention_list = [attention_mask, attention_mask_0, attention_mask_1, attention_mask_2]
        input_ids = [x for x in potential_input_list if x is not None]
        attention_mask = [x for x in potential_attention_list if x is not None]
        num_input_ids = len(input_ids)
        batch_size = input_ids[0].size(0)
        # compute proxy score
        proxy_input_ids = torch.concat(input_ids, dim=0)
        proxy_attention_mask = torch.concat(attention_mask, dim=0)
        proxy_outputs = self.proxy(
            proxy_input_ids,
            attention_mask=proxy_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # del proxy_input_ids
        # del proxy_attention_mask
        # get logits for each token
        proxy_outputs = proxy_outputs[0]
        # proxy_outputs -> [bs, seq_len, 2]
        if inference:
            # handle weight all equals to zero
            tmp_max = torch.nn.functional.one_hot(proxy_outputs[:, :, 0].argmax(dim=-1), proxy_outputs.size(1)) # [bs]
            proxy_outputs = (proxy_outputs.argmax(dim=-1)==0).to(torch.float) # [bs, seq_len]
            equal_to_zero = (proxy_outputs.sum(dim=-1) == 0).to(torch.float) # [bs]
            proxy_outputs += equal_to_zero.unsqueeze(-1) * tmp_max 
            
        else:
            tau = max(self.min_tau, math.exp(-self.t * self.exp_scheduler_hyper))
            self.t += batch_size
            proxy_outputs = torch.nn.functional.gumbel_softmax(proxy_outputs, tau=tau)[:, :, 0]


        # for CL
        input_ids = input_ids * 2 if num_input_ids == 1 and not inference else input_ids
        proxy_outputs = torch.concat((proxy_outputs, proxy_outputs)) if num_input_ids == 1 and not inference else proxy_outputs
        input_ids = torch.concat(input_ids, dim=0)
        attention_mask = attention_mask * 2 if num_input_ids == 1 and not inference else attention_mask
        attention_mask = torch.concat(attention_mask, dim=0)
        for layer in self.model.encoder.layer:
            setattr(layer.attention.self, "proxy_outputs", proxy_outputs)

        outputs = self.model(
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
        for layer in self.model.encoder.layer:
            delattr(layer.attention.self, "proxy_outputs")
        pooler_output  = self.pooler(attention_mask, outputs, proxy_outputs=proxy_outputs)
        pooler_output = self.simple_head(pooler_output)
        anchor_output = pooler_output[:batch_size]
        positive_and_negative_output = pooler_output[batch_size:]
        cos_sim = self.similarity(anchor_output.unsqueeze(1), positive_and_negative_output.unsqueeze(0)) if not inference else None
        labels = torch.arange(batch_size, dtype=torch.long, device=anchor_output.device) if not inference else None

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