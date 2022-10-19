import math
from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertLayer,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput
)
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
    
class GumbelSoftmaxBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=False):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = GumbelSoftmaxBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
    # def forward(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     token_type_ids: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     head_mask: Optional[torch.Tensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], ContrastiveLearningOutput]:
    #     pass
class GumbelSoftmaxBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([GumbelSoftmaxBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
class GumbelSoftmaxBertLayer(BertLayer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GumbelSoftmaxBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
class GumbelSoftmaxBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super(BertAttention, self).__init__()
        self.self = GumbelSoftmaxBertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
class GumbelSoftmaxBertSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        # proxy_outputs: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        proxy_outputs = getattr(self, "proxy_outputs", None)
        mixed_query_layer = self.query(hidden_states) # [bs, seq_len, hidden]

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # # [bs, seq_len, num_head, head_size]
        #     x = x.view(new_x_shape)
        #     return x.permute(0, 2, 1, 3)
        # # [bs, num_head, seq_len, head_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, num_head, seq_len, seq_len]

        # relative key is not implemented
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # attention_probs -> [bs, num_head, seq_len, seq_len]
        # final dim -> attn prob for each token in sentence
        # \times `proxy_output` in last dim <-> gumbel softmax on whether the token is seen
        # THIS IS THE MOST MEANINGFUL PLACE TO INTRODUCE A PROBABILITY
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        if proxy_outputs is not None:
            attention_probs = attention_probs * proxy_outputs[:, None, None, :]
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs