from .define import M
import torch
import torch.nn as nn

class AddBias(M):
    def __init__(self):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor(1))

    def forward(self, x):
        return x + self.bias