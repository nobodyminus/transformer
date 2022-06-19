import torch


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_mask = torch.tril(torch.ones(attn_shape),
                          diagonal=1).type(torch.uint8)
    return sub_mask
