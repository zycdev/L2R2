import torch.nn as nn
from torch.nn import functional as F

from transformers import (BertPreTrainedModel, BertModel,
                          RobertaModel, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)


class BertForListRank(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., list_len]`` where `list_len` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, list_len)`` where `list_len` is the size of
            the second dimension of the input tensors. (see `input_ids` above).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, seq_len, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, seq_len, seq_len)``:
            Attentions weights after softmax, used to compute the weighted average in the self-attention heads.
    """

    def __init__(self, config):
        super(BertForListRank, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        batch_size, list_len, seq_len = input_ids.shape

        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)

            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]

        outputs = self.bert(flat_input_ids,
                            position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)

        # pooled_output = outputs[1]
        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).squeeze(-1)  # (real_seq_num,)

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])
            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        # (batch_size, list_len)
        logits = logits.view(batch_size, list_len)

        return logits


class RobertaForListRank(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., list_len]`` where `list_len` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, list_len)`` where `list_len` is the size of
            the second dimension of the input tensors. (see `input_ids` above).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, seq_len, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, seq_len, seq_len)``:
            Attentions weights after softmax, used to compute the weighted average in the self-attention heads.
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForListRank, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        batch_size, list_len, seq_len = input_ids.shape

        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)

            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]

        '''
        chunk_size = 24
        chunk_input_ids = flat_input_ids.split(chunk_size, dim=0)
        if flat_position_ids is not None:
            chunk_position_ids = flat_position_ids.split(chunk_size, dim=0)
        else:
            chunk_position_ids = [None] * len(chunk_input_ids)
        if flat_token_type_ids is not None:
            chunk_token_type_ids = flat_token_type_ids.split(chunk_size, dim=0)
        else:
            chunk_token_type_ids = [None] * len(chunk_input_ids)
        if flat_attention_mask is not None:
            chunk_attention_mask = flat_attention_mask.split(chunk_size, dim=0)
        else:
            chunk_attention_mask = [None] * len(chunk_input_ids)
        chunk_hidden_states = []
        for i in range(len(chunk_input_ids)):
            outputs = self.roberta(chunk_input_ids[i],
                                   position_ids=chunk_position_ids[i],
                                   # token_type_ids=chunk_token_type_ids[i],
                                   attention_mask=chunk_attention_mask[i], head_mask=head_mask)
            chunk_hidden_states.append(outputs[0])  # (chunk_size, max_seq_len, hidden_size)
        hidden_states = torch.cat(chunk_hidden_states, dim=0)  # (real_seq_num, max_seq_len, hidden_size)
        '''

        outputs = self.roberta(flat_input_ids,
                               position_ids=flat_position_ids,
                               # token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)

        # pooled_output = outputs[1]
        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).squeeze(-1)  # (real_seq_num,)

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])
            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        # (batch_size, list_len)
        logits = logits.view(batch_size, list_len)

        return logits
