from copy import deepcopy as c

import torch.nn as nn
import torch.nn.functional as f

from model.decoder import Decoder, DecoderLayer
from model.encoder import Encoder, EncoderLayer
from model.layers import MultiLayerAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return f.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, mem, src_mask, tgt, tgt_msk):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_msk)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt,
                           tgt_mask)


def make_model(src_voc, tgt_voc, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiLayerAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
                           nn.Sequential(Embeddings(d_model, src_voc), c(pos)),
                           nn.Sequential(Embeddings(d_model, tgt_voc), c(pos)),
                           Generator(d_model, tgt_voc))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
