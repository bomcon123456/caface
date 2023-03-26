import torch
import torch.nn as nn


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz // 2
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
