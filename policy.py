# policy.py

import torch
import torch.nn as nn
import torch.distributions as D

class PolicyNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.mean = nn.Linear(64, 3)
        self.log_std = nn.Parameter(torch.zeros(3))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return D.Normal(mean, std)
