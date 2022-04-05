import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import params
from cuda import use_cuda
from env.operator import num_operators


class PointerHead(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.lin = nn.Linear(emb_size, emb_size)
        self.vec = nn.Linear(emb_size, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]

        Returns: probability distribution, shape [batch_size, seq_len]
        """
        return self.vec(F.selu(self.lin(x))).squeeze(dim=2)


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
            nn.TransformerEncoderLayer(params.var_encoder_size, 8, 1024, activation=F.selu, batch_first=True), 5)
        self.decoder = nn.Linear(params.var_encoder_size, params.dense_output_size)

    def forward(self, x):
        x, num_batches = self.embed_state(x)
        x = F.selu(self.var_encoder(x)).view(num_batches * params.num_examples, params.state_len, -1)
        x = self.pos_encoding(x)
        x = self.transformer(x)  # x: [num_batches * params.num_examples, params.state_len, params.var_encoder_size]
        x = F.selu(self.encoder(x))
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
        self.operator_head = nn.Linear(params.dense_output_size, num_operators)
        self.first_variable_head = PointerHead(params.dense_output_size)
        self.second_variable_head = PointerHead(params.dense_output_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.operator_head(x), self.first_variable_head(x), self.second_variable_head(x)

    def predict(self, x):
        statement_pred, drop_pred, _ = self.forward(x)
        statement_probs = F.softmax(statement_pred, dim=-1).data
        drop_indx = np.argmax(drop_pred.data.cpu().numpy(), axis=-1)
        return np.argsort(statement_probs.cpu().numpy()), statement_probs, drop_indx