import torch
import torch.nn as nn

from backbones.grl import GradientReversal

class Discriminator(nn.Module):
    def __init__(self, n_domains, input_shape, bottleneck_shape=512, drop_out=0.5, activation_fnc="leaky"):
        super(Discriminator, self).__init__()

        if activation_fnc == "leaky":
            activation_fnc = nn.LeakyReLU
        else:
            activation_fnc = nn.ReLU

        self.grl = GradientReversal(),
        self.model = nn.Sequential(
            nn.Linear(input_shape, bottleneck_shape),
            activation_fnc(),
            nn.Dropout(drop_out),
            nn.Linear(bottleneck_shape, n_domains)
        )

        self.model.apply(self.init_fc)
    
    @staticmethod
    def init_fc(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)
 
    def forward(self, x, coeff=1):
        x = self.grl(x, coeff)
        return self.model(x)