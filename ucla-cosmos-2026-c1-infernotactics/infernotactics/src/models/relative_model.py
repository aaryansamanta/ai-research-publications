"""Model used by the v8 fire-relative action experiment.

Unlike the historical actor, the target head scores semantic candidates for
the selected resource.  It never emits an absolute zone class.
"""

import torch
import torch.nn as nn
import numpy as np

from models.cnn_branch import CNNBranch, PER_CELL_CHANNELS, POOLED_DIM
from models.classification_head import ClassificationHead
from models.mlp_branch import MLPBranch, OUTPUT_DIM as MLP_OUTPUT_DIM
from train.relative_actions import N_TARGET_TYPES, TARGET_FEATURE_DIM
from env.inferno_env import flatten_scalars


class RelativeInfernoModel(nn.Module):
    def __init__(self, n_grid_channels, n_scalars, n_resources, n_zones, hidden_dim=128):
        super().__init__()
        self.cnn = CNNBranch(in_channels=n_grid_channels)
        self.mlp = MLPBranch(in_features=n_scalars)
        self.trunk = nn.Sequential(
            nn.Linear(POOLED_DIM + MLP_OUTPUT_DIM, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resource_head = nn.Linear(hidden_dim, n_resources)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.resource_embedding = nn.Embedding(n_resources, 16)
        self.target_head = nn.Sequential(
            nn.Linear(MLP_OUTPUT_DIM + PER_CELL_CHANNELS + TARGET_FEATURE_DIM + 16, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 1),
        )
        self.classifier = ClassificationHead(in_channels=PER_CELL_CHANNELS)
        self.n_resources = n_resources
        self.n_zones = n_zones

    @staticmethod
    def obs_to_tensors(obs, device=None):
        grid = torch.from_numpy(np.ascontiguousarray(obs["grid"])).unsqueeze(0)
        scalars = torch.from_numpy(flatten_scalars(obs["scalars"])).unsqueeze(0)
        if device is not None:
            grid = grid.to(device)
            scalars = scalars.to(device)
        return grid, scalars

    def forward(self, grid, scalars, target_zones, target_features):
        per_cell, pooled_cnn, zone_pooled = self.cnn(grid)
        mlp_features = self.mlp(scalars)
        fused = torch.cat([pooled_cnn, mlp_features], dim=1)
        hidden = self.trunk(fused)
        resource_logits = self.resource_head(hidden)
        value = self.value_head(hidden)

        # target_zones: (B, resources, candidates); invalid candidates use -1.
        # Gather target-zone features with a one-hot matmul instead of
        # torch.gather: DirectML's scatter-backward can't handle a gather
        # whose index dim (candidates) differs from the input dim (zones),
        # while a (B,R,C,Z) @ (B,R,Z,F) matmul is fully supported.
        safe_zones = target_zones.clamp(min=0)
        n_zones = zone_pooled.shape[2]
        zone_ids = torch.arange(n_zones, device=grid.device)
        onehot_zones = (safe_zones.unsqueeze(-1) == zone_ids).float()
        zone_pooled = zone_pooled.unsqueeze(1).expand(-1, self.n_resources, -1, -1)
        zone_features = torch.matmul(onehot_zones, zone_pooled)
        global_features = mlp_features.unsqueeze(1).unsqueeze(1).expand(
            -1, self.n_resources, target_features.shape[2], -1
        )
        resource_ids = torch.arange(self.n_resources, device=grid.device)
        resource_embed = self.resource_embedding(resource_ids).view(1, self.n_resources, 1, -1)
        resource_embed = resource_embed.expand(grid.shape[0], -1, target_features.shape[2], -1)
        target_input = torch.cat([global_features, zone_features, target_features, resource_embed], dim=-1)
        target_logits = self.target_head(target_input).squeeze(-1)
        target_logits = target_logits.masked_fill(target_zones < 0, -1e9)

        classification_logits = self.classifier(per_cell)
        return {"resource_type": resource_logits, "target": target_logits}, value, classification_logits
