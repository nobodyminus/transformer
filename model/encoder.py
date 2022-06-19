import torch.nn as nn

from model.layers import LayerNorm, SublayerConnection
from model.utils import clone


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, dense, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.dense = dense
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda l: self.attn(l, l, l, mask))
        return self.sublayer[1](x, self.dense)


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clone(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
