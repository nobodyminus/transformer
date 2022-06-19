from torch import nn as nn
from model.utils import clone
from model.layers import LayerNorm, SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, dense, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.dense = dense
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, mem, src_msk, tgt_msk):
        x = self.sublayer[0](x, lambda l: self.self_attn(l, l, l, tgt_msk))
        x = self.sublayer[0](x, lambda l: self.self_attn(l, mem, mem, src_msk))
        return self.sublayer[2](x, self.dense)


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clone(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mem, src_msk, tgt_msk):
        for layer in self.layers:
            x = layer(x, mem, src_msk, tgt_msk)
        return self.norm(x)
