import torch.nn as nn
from copy import deepcopy


def clone(module, number):
    return nn.ModuleList([deepcopy(module) for _ in range(number)])
