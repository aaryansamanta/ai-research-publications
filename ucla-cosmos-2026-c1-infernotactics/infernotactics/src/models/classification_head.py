"""
Auxiliary per-cell classification head (one of the 4 required ML concepts):
takes the CNN branch's full-resolution per-cell feature map and predicts a
4-class fire state per cell (Safe / Fuel / Threat / Blaze).

Ground truth is literally already in the environment: fire_sim.FireSim.state
(the last channel of the 'grid' observation) is 0=Safe/1=Fuel/2=Threat/
3=Blaze/4=Burned Out -- 5 raw values. This head collapses Burned Out into
the Safe class for the classification target (fire_state_to_class below):
by the time a cell is burned out its fuel is spent, so it behaves like a
Safe cell for everything that matters tactically (can't ignite or
re-ignite). This matches the 4-class Safe/Fuel/Threat/Blaze head from the
project plan rather than adding a 5th class for that essentially-inert
terminal state. Flag/revisit if the project plan actually wants Burned Out
distinguished from Safe.

Training this head (cross-entropy against fire_state) is the classification
piece of the 4-concept fusion -- NOT wired into a training loop yet.
"""

import torch
import torch.nn as nn

N_CLASSES = 4
CLASS_NAMES = ("Safe", "Fuel", "Threat", "Blaze")

# fire_sim.py state ordinals (SAFE=0, FUEL=1, THREAT=2, BLAZE=3, BURNED_OUT=4)
# -> this head's class indices (BURNED_OUT folds into SAFE, see module docstring)
_STATE_TO_CLASS = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)


def fire_state_to_class(fire_state):
    """fire_state: integer tensor of raw fire_sim ordinals (0-4, any shape)
    -> same-shape tensor of this head's 4 class indices. Use this to build
    ground-truth labels for the auxiliary cross-entropy loss once training
    starts."""
    return _STATE_TO_CLASS.to(fire_state.device)[fire_state.long()]


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, n_classes=N_CLASSES):
        super().__init__()
        # 1x1 conv: a per-cell linear classifier over the CNN's per-cell
        # feature map -- preserves (H, W) exactly. No extra receptive-field
        # growth needed here since per_cell_trunk already mixed in local
        # context via its two 3x3 convs.
        self.classifier = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, per_cell_features):
        """per_cell_features: (B, in_channels, H, W) -> logits (B, n_classes, H, W)"""
        return self.classifier(per_cell_features)
