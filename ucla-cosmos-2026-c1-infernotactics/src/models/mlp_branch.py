"""
MLP branch: consumes the InfernoEnv 'scalars' observation (wind speed/
direction, humidity, resources available per type, elapsed ticks --
env.inferno_env.flatten_scalars() gives the fixed-order vector) and produces
a feature vector sized to match the CNN branch's pooled output, so fusion is
a plain concat of two comparable-sized parts.
"""

import torch.nn as nn

OUTPUT_DIM = 128  # matches cnn_branch.POOLED_DIM


class MLPBranch(nn.Module):
    def __init__(self, in_features, output_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, scalars):
        """scalars: (B, in_features) -> (B, output_dim)"""
        return self.net(scalars)
