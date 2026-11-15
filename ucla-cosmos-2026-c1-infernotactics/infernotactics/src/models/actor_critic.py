"""
Actor-critic head (the RL piece of the 4-concept fusion): takes the fused
CNN+MLP feature vector and outputs an action distribution plus a state-value
estimate.

Action factorization
---------------------
InfernoEnv's action is (resource_type, target_zone). Rather than one flat
categorical over all 3*32=96 combinations, this factors the actor into two
independent categorical heads:

    p(resource_type, target_zone) = p(resource_type) * p(target_zone)

i.e. resource_type and target_zone are sampled independently, not
autoregressively (the zone logits do not condition on the sampled resource
type). This is the simpler of the two standard options for a
MultiDiscrete-style action space and a reasonable starting point -- revisit
only if training shows the two need to interact (e.g. a resource type
systematically wanting different zones than the others).

Note this head always proposes a (resource_type, zone) pair; it does not
have a separate "noop" logit even though InfernoEnv.step() accepts one --
matches the factorization specified in the project plan as-is.
"""

import torch
import torch.nn as nn

N_RESOURCE_TYPES = 4  # water_team, trench_crew, rescue_vehicle, helicopter
N_ZONES = 32
HIDDEN_DIM = 128
ZONE_FEATURE_DIM = 32  # per-zone channel count from cnn_branch.CNNBranch's zone_pooled output
ZONE_HEAD_HIDDEN = 64
# Dimension of the RAW MLP branch output the zone head reads (see ZoneHead's
# docstring for why this is no longer HIDDEN_DIM/the shared trunk output).
# Kept as its own constant rather than importing models.mlp_branch.OUTPUT_DIM
# here, matching the existing convention that ZONE_FEATURE_DIM above is also
# a locally-duplicated constant rather than an import from cnn_branch -- this
# file stays decoupled from the other model files; the two numbers must
# match models.mlp_branch.OUTPUT_DIM (currently 128) by convention, not import.
MLP_FEATURE_DIM = 128


class ZoneHead(nn.Module):
    """Gives the zone head direct access to each zone's OWN pooled spatial
    features (from cnn_branch.CNNBranch's zone_pooled), concatenated with
    the RAW global MLP vector -- NOT the shared post-trunk hidden state `h`
    that resource_type_head/value_head read. Applies the SAME small shared
    MLP independently to each of the n_zones (global_features,
    per_zone_features) pairs -- weight-tied across zones (no reason zone 3
    vs zone 19 should need separate parameters), producing one logit per
    zone; this per-zone weight-sharing already existed in the first version
    of this head.

    What changed and why: a diagnostic (diagnostic_zonehead_dependence.py,
    checkpoint models/checkpoints_zonehead_zonehead_v1/episode_0300.pt)
    found zone_pooled itself carries excellent, correctly zone-specific fire
    signal (r=0.97 vs real per-zone Threat+Blaze counts across 32 zones),
    but the zone head's OUTPUT barely varied with the observation (variance
    26x below the untouched resource_type head's) and did not track real
    fire location (r=-0.05 to -0.11). One candidate explanation: `h =
    trunk(fused)` mixes the (separately confirmed near-constant) globally-
    pooled CNN vector together with the MLP output through ONE Linear layer
    shared by all three heads -- optimized jointly for resource_type_head's
    and value_head's needs too, not just this head's. Feeding the zone head
    the raw MLP vector directly, bypassing that shared bottleneck, removes
    one place the per-zone signal could be getting diluted/entangled before
    the zone head ever uses it. This is one candidate fix, not a proven
    mechanism -- see train_zonehead_fix.py's direct test."""

    def __init__(self, global_dim, per_zone_dim=ZONE_FEATURE_DIM, n_zones=N_ZONES, hidden=ZONE_HEAD_HIDDEN):
        super().__init__()
        self.n_zones = n_zones
        self.net = nn.Sequential(
            nn.Linear(global_dim + per_zone_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_features, zone_features):
        """global_features: (B, global_dim), zone_features: (B, n_zones, per_zone_dim)
        -> zone logits (B, n_zones)"""
        global_expanded = global_features.unsqueeze(1).expand(-1, self.n_zones, -1)
        combined = torch.cat([global_expanded, zone_features], dim=-1)
        return self.net(combined).squeeze(-1)


class ActorCritic(nn.Module):
    """n_value_heads=1 (default) is byte-for-byte the original architecture --
    a single self.value_head attribute, forward()'s value_head_idx argument
    unused -- so every existing caller (train_actor_critic.py,
    train_actor_critic_multi.py/v2, heuristic_policy.py, eval.py) is
    completely unaffected. n_value_heads>1 (added for
    train_actor_critic_multi_v3.py's separate-value-head-per-scenario
    experiment) instead builds self.value_heads, an nn.ModuleList, and
    forward()'s value_head_idx selects which one to use for a given
    call -- the shared trunk/actor heads are untouched either way, only the
    final value projection becomes scenario-specific. See
    train_actor_critic_multi_v3.py's module docstring for why (per-scenario
    return normalization alone, in v2, wasn't sufficient -- the shared value
    head's LAST layer was still one set of weights receiving gradient from
    every scenario's wildly different-scale value loss every episode)."""

    def __init__(self, in_features, n_resource_types=N_RESOURCE_TYPES, n_zones=N_ZONES, hidden_dim=HIDDEN_DIM,
                 n_value_heads=1, zone_feature_dim=ZONE_FEATURE_DIM, mlp_feature_dim=MLP_FEATURE_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resource_type_head = nn.Linear(hidden_dim, n_resource_types)
        # ZoneHead reads the RAW MLP vector (mlp_feature_dim-wide) plus each
        # zone's own pooled spatial features (from cnn_branch.CNNBranch's
        # zone_pooled output) -- NOT the shared trunk context h that
        # resource_type_head/value_head use. See ZoneHead's docstring for why.
        self.zone_head = ZoneHead(mlp_feature_dim, per_zone_dim=zone_feature_dim, n_zones=n_zones)
        self.n_value_heads = n_value_heads
        if n_value_heads == 1:
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            self.value_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_value_heads)])

    def forward(self, fused, zone_features, mlp_features, value_head_idx=0):
        """fused: (B, in_features) -- feeds the shared trunk (resource_type_head,
            value_head only).
        zone_features: (B, n_zones, zone_feature_dim) -- per-zone pooled CNN features.
        mlp_features: (B, mlp_feature_dim) -- RAW MLP branch output (pre-trunk),
            what the zone head reads instead of the shared trunk context.
        -> ({'resource_type': (B, n_resource_types) logits, 'zone': (B, n_zones) logits},
            value: (B, 1))
        value_head_idx is ignored when n_value_heads==1 (the original,
        single-shared-head behavior)."""
        h = self.trunk(fused)
        action_logits = {
            "resource_type": self.resource_type_head(h),
            "zone": self.zone_head(mlp_features, zone_features),
        }
        value = self.value_head(h) if self.n_value_heads == 1 else self.value_heads[value_head_idx](h)
        return action_logits, value
