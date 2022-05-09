import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import params
from cuda import use_cuda
from transformer.statement import num_incomplete_statements, incomplete_statement_to_index, parse_args, \
    statement_to_operator, statement_to_variables, statement_to_variables_mask
from env.statement import num_statements, index_to_statement

from typing import List


class Cache:
    def __init__(self):
        self.cache = None
        self.cache_mode = False
        self.cache_children = []

    def register_children(self, children):
        self.cache_children = children

    def set_mode(self, cache_mode):
        self.cache_mode = cache_mode
        for child in self.cache_children:
            child.set_mode(cache_mode)

    def clear_cache(self):
        self.cache = None
        for child in self.cache_children:
            child.clear_cache()

    def update_cache(self, x, seq_dim):
        if self.cache is None:
            self.cache = x
        else:
            self.cache = torch.cat([self.cache, x], seq_dim)

    def data_slice(self, mask, seq_dim):
        cache_len = 0 if self.cache is None else self.cache.shape[seq_dim]
        causal_line = mask[cache_len][cache_len:]

        start = torch.argmax(causal_line) + cache_len
        end = torch.argmin(causal_line) + cache_len

        return start, end

    def resample_cache(self, batch_indices):
        if self.cache is not None:
            initial_shape = self.cache.shape
            self.cache = self.cache.view(len(batch_indices), -1)[batch_indices]
            self.cache = self.cache.reshape(initial_shape)
        for child in self.cache_children:
            child.resample_cache(batch_indices)


class PointerHead(nn.Module, Cache):
    def __init__(self, emb_size):
        nn.Module.__init__(self)
        Cache.__init__(self)
        self.w_query = nn.Linear(emb_size, emb_size)
        self.w_key = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            mask: Tensor, shape [seq_len, seq_len]

        Returns: probability distribution, shape [batch_size, seq_len, seq_len]
        """
        emb_size = x.shape[-1]
        key = self.w_key(x)

        if self.cache_mode:
            start, end = self.data_slice(mask, 1)
            query = self.w_query(x[:, start:end])
            scores = torch.bmm(query, key.transpose(-2, -1))
        else:
            query = self.w_query(x)
            if mask is None:
                scores = torch.bmm(query, key.transpose(-2, -1))
            else:
                scores = torch.baddbmm(mask, query, key.transpose(-2, -1))

        scores /= emb_size ** (1 / 2)

        if self.cache_mode:
            batch_len, seq_len, var_len = scores.shape
            scores = torch.cat([scores, torch.ones(batch_len, seq_len, params.state_len - var_len) * float('-inf')],
                               dim=-1)
            self.update_cache(scores, 1)
            return self.cache
        else:
            return scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CacheEncoderLayer(nn.TransformerEncoderLayer, Cache):
    def __init__(self, d_model, n_head, dim_feedforward, batch_first=True):
        nn.TransformerEncoderLayer.__init__(self, d_model, n_head, dim_feedforward, batch_first=batch_first)
        Cache.__init__(self)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        if self.cache_mode:
            start, end = self.data_slice(src_mask, 1)
            x = self.self_attn(x[:, start:end], x, x,
                               attn_mask=None,
                               key_padding_mask=src_key_padding_mask,
                               need_weights=False)[0]
            x = self.norm1(x + self.dropout1(x))
            x = self.norm2(x + self._ff_block(x))
            self.update_cache(x, 1)
            return self.cache
        else:
            return super().forward(x, src_mask, src_key_padding_mask)


class CacheTransformerEncoder(nn.TransformerEncoder, Cache):
    def __init__(self, encoder_layer, num_layers):
        nn.TransformerEncoder.__init__(self, encoder_layer, num_layers)
        Cache.__init__(self)
        self.register_children(self.layers)


class Encoder(nn.Module, Cache):
    def __init__(self):
        nn.Module.__init__(self)
        Cache.__init__(self)

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.pos_encoding = PositionalEncoding(params.var_encoder_size)

        self.transformer = CacheTransformerEncoder(
            CacheEncoderLayer(params.var_encoder_size, 8, 1024, batch_first=True), 5)
        self.encoder = nn.Linear(params.var_encoder_size, params.dense_output_size)

        self.register_children([self.transformer])

    def forward(self, x, mask):
        x, num_batches = self.embed_state(x)
        x = F.relu(self.var_encoder(x)).view(num_batches * params.num_examples, params.state_len, -1)
        x = self.pos_encoding(x)

        # x: [num_batches * params.num_examples, state_len, params.var_encoder_size]
        x = self.transformer(x, mask=mask)
        x = F.relu(self.encoder(x))
        x = x.view(num_batches, params.num_examples, -1, params.dense_output_size).mean(dim=1)

        return x  # x: [num_batches, state_len, params.dense_output_size]

    def embed_state(self, x):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, params.num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches


class BothWiseEncoder(nn.Module, Cache):
    def __init__(self):
        nn.Module.__init__(self)
        Cache.__init__(self)

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.pos_encoding = PositionalEncoding(params.var_encoder_size)

        self.word_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(params.var_encoder_size, 8, 1024, batch_first=True), 5)
        self.example_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(params.var_encoder_size, 8, 1024, batch_first=True), 3)
        self.encoder = nn.Linear(params.var_encoder_size, params.dense_output_size)

        self.register_children([self.word_transformer, self.example_transformer])

    def forward(self, x, mask):
        x, num_batches = self.embed_state(x)
        x = F.relu(self.var_encoder(x)).view(num_batches * params.num_examples, params.state_len, -1)
        x = self.pos_encoding(x)

        # x: [num_batches * params.num_examples, params.state_len, params.var_encoder_size]
        x = self.word_transformer(x, mask=mask)

        x = torch.transpose(x.view(num_batches, params.num_examples, params.state_len, params.var_encoder_size), 1,
                            2).reshape(num_batches * params.state_len, params.num_examples, params.var_encoder_size)
        # x: [num_batches * params.state_len, params.num_examples, params,var_encoder_size]
        x = self.example_transformer(x)
        x = torch.transpose(x.view(num_batches, params.state_len, params.num_examples, params.var_encoder_size), 1,
                            2)

        x = F.relu(self.encoder(x))
        x = x.mean(dim=1)

        return x  # x: [num_batches, params.state_len, params.dense_output_size]

    def embed_state(self, x):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, params.num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches


class BaseModel(nn.Module):
    def load(self, path):
        if use_cuda:
            par = torch.load(path)
        else:
            par = torch.load(path, map_location=lambda storage, loc: storage)

        state = self.state_dict()
        for name, val in par.items():
            if name in state:
                assert state[name].shape == val.shape, "%s size has changed from %s to %s" % \
                                                       (name, state[name].shape, val.shape)
                state[name].copy_(val)
            else:
                print("WARNING: %s not in model during model loading!" % name)

    def save(self, path):
        torch.save(self.state_dict(), path)


class PCCoder(BaseModel, Cache):
    def __init__(self):
        BaseModel.__init__(self)
        Cache.__init__(self)

        self.encoder = Encoder()
        self.operator_head = nn.Linear(params.dense_output_size, num_incomplete_statements + 1)
        self.variables_head = nn.ModuleList([PointerHead(params.dense_output_size)
                                             for _ in range(params.num_variable_head)])

        self.register_children([self.encoder] + [var_head for var_head in self.variables_head])

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        return (self.operator_head(x),) + tuple(head(x, mask) for head in self.variables_head)

    def get_prob(self, x, num_inputs, num_vars):
        operator_pred, *variables_pred = self.forward(x, generate_mask())

        batch_indices = torch.arange(len(num_inputs))
        operator_pred = logit_to_log_prob(operator_pred[batch_indices, params.num_inputs + (num_vars - num_inputs)])
        variables_pred = [var_pred[batch_indices, params.num_inputs + (num_vars - num_inputs)]
                          for var_pred in variables_pred]

        index_mask = torch.arange(params.state_len).repeat(len(num_inputs), 1)
        zero_mask = (index_mask >= num_inputs[:, None]) & (index_mask <= params.num_inputs)
        for i in range(params.num_variable_head):
            variables_pred[i][zero_mask] = -torch.inf
            variables_pred[i] = logit_to_log_prob(variables_pred[i])

        return operator_pred, variables_pred

    def get_statement_prob_slow(self, num_inputs, operator_pred, variables_pred):
        batch_indices = torch.arange(len(num_inputs))
        statement_log_probs = torch.zeros(len(batch_indices), num_statements)

        for i, statement in index_to_statement.items():
            func, args = statement.function, statement.args
            incomplete_statement, variables, variables_mask = parse_args(func, args, num_inputs)
            operator_index = incomplete_statement_to_index[incomplete_statement]

            statement_log_probs[:, i] = operator_pred[:, operator_index]
            for j in range(params.num_variable_head):
                if isinstance(variables[j], int):
                    if variables[j] >= params.state_len:
                        statement_log_probs[:, i] -= 1e6
                    else:
                        statement_log_probs[:, i] += variables_mask[j] * variables_pred[j][:, variables[j]]
                else:
                    overflow_mask = variables[j] >= params.state_len
                    variables[j] = torch.clamp(variables[j], max=params.state_len - 1)
                    statement_log_probs[:, i] += variables_mask[j] * variables_pred[j][batch_indices, variables[j]]
                    statement_log_probs[:, i] -= 1e6 * overflow_mask

        return statement_log_probs

    def get_statement_prob(self, num_inputs, operator_pred, variables_pred):
        batch_indices = torch.arange(len(num_inputs))
        statement_log_probs = operator_pred[batch_indices[:, None], statement_to_operator[None, :]]
        statement_to_variables_: List[torch.Tensor] = [x[None, :] for x in statement_to_variables]

        for j in range(params.num_variable_head):
            statement_to_variables_[j] = torch.where(statement_to_variables_[j] < num_inputs[:, None],
                                                     statement_to_variables_[j],
                                                     statement_to_variables_[j] + 1 + (
                                                             params.num_inputs - num_inputs[:, None]))

            overflow_mask = statement_to_variables_[j] >= params.state_len
            statement_to_variables_[j] = torch.clamp(statement_to_variables_[j], max=params.state_len - 1)
            statement_log_probs += statement_to_variables_mask[j][None, :] * \
                                   variables_pred[j][batch_indices[:, None], statement_to_variables_[j]]
            statement_log_probs -= 1e6 * overflow_mask

        return statement_log_probs

    def predict(self, x, num_inputs, num_vars):
        with torch.no_grad():
            operator_pred, variables_pred = self.get_prob(x, num_inputs, num_vars)
            statement_log_probs = self.get_statement_prob(num_inputs, operator_pred, variables_pred)

        return np.argsort(statement_log_probs.detach().numpy()), statement_log_probs


def generate_mask():
    mask = torch.zeros(params.state_len, params.state_len)
    mask[:params.num_inputs + 1, params.num_inputs + 1:] = float('-inf')
    mask[params.num_inputs + 1:, params.num_inputs + 1:] = \
        torch.triu(torch.ones(params.max_program_len, params.max_program_len) * float('-inf'), diagonal=1)
    return mask


def logit_to_log_prob(x):
    return torch.clip(F.log_softmax(x, dim=1), min=-1e6)
