# deep_optimal_stopping/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class StoppingRuleNet(nn.Module):
    """Neural network for optimal stopping decisions."""

    def __init__(self, d, q1, q2):
        super().__init__()
        self.hidden1 = nn.Linear(d, q1)
        self.batch1  = nn.BatchNorm1d(q1)
        self.hidden2 = nn.Linear(q1, q2)
        self.batch2  = nn.BatchNorm1d(q2)
        self.output  = nn.Linear(q2, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = F.relu(self.batch1(self.hidden1(x)))
        x = F.relu(self.batch2(self.hidden2(x)))
        return torch.sigmoid(self.output(x))
