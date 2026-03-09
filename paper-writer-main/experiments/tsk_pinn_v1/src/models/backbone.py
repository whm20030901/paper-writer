import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, depth=4, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h), h
