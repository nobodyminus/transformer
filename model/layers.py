import math

import torch
from torch import nn as nn

from model.utils import clone


class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.epsilon) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiLayerAttention(nn.Module):
    def __init__(self, h, dmodel, dropout=0.1):
        super(MultiLayerAttention, self).__init__()
        assert dmodel % h == 0
        self.dk = dmodel // h
        self.h = h
        self.dense = clone(nn.Linear(dmodel, dmodel), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = q.size(0)
        q, k, v = [lin(x).view(n_batches, -1, self.h, self.dk).transpose(1, 2)
                   for lin, x in zip(self.dense, (q, k, v))]
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = (x.transpose(1, 2).contiguous().view(n_batches, -1,
                                                 self.h * self.dk))

        return self.dense[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


def attention(q, k, v, mask=None, dropout=None):
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn

