import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import params
from cuda import use_cuda
from transformer.statement import num_incomplete_statements


class PointerHead(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
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
        query = self.w_query(x)
        key = self.w_key(x)

        if mask is None:
            scores = torch.bmm(query, key.transpose(-2, -1))
        else:
            scores = torch.baddbmm(mask, query, key.transpose(-2, -1))

        return scores / emb_size ** (1 / 2)


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
        x = x + self.pe[:x.size(0)].transpose(0, 1)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.pos_encoding = PositionalEncoding(params.var_encoder_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(params.var_encoder_size, 8, 1024, batch_first=True), 5)
        self.encoder = nn.Linear(params.var_encoder_size, params.dense_output_size)

    def forward(self, x, mask):
        x, num_batches = self.embed_state(x)
        x = F.relu(self.var_encoder(x)).view(num_batches * params.num_examples, params.state_len, -1)
        x = self.pos_encoding(x)

        # x: [num_batches * params.num_examples, params.state_len, params.var_encoder_size]
        x = self.transformer(x, src_mask=mask)
        x = F.relu(self.encoder(x))
        x = x.view(num_batches, params.num_examples, params.state_len, -1).mean(dim=1)

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


class PCCoder(BaseModel):
    def __init__(self):
        super(PCCoder, self).__init__()
        self.encoder = Encoder()
        self.operator_head = nn.Linear(params.dense_output_size, num_incomplete_statements + 1)
        self.variables_head = nn.ModuleList([PointerHead(params.dense_output_size)
                                             for _ in range(params.num_variable_head)])

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        return (self.operator_head(x),) + tuple(head(x, mask) for head in self.variables_head)

    def predict(self, x):
        pass


def generate_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    mask[:, -1] = 0
    return mask
