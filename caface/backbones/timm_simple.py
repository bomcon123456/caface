
from enum import Enum
import sys


import timm
import torch
import torch.nn as nn


class TIMMSimple(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        timm_dict = {
            **config
        }
        print("Timm dict:", timm_dict)
        self.model = timm.create_model(**timm_dict)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    from configs.custom_configs.timmsim.base import config
    from rich import print

    print(config.timm)
    m = TIMMSimple(config.timm)
    input = torch.rand((4, 3, 112, 112))
    f = m(input)
    print(f.shape)
