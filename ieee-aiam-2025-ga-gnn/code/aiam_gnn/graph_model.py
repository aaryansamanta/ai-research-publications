"""Per-modality feature graphs, graph encoders and learned probability fusion."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def feature_graph(train, neighbors):
    """Symmetric union of top-k absolute training correlations, with self-loops."""
    centered = train - train.mean(axis=0)
    norms = np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
    unit = centered / norms
    similarity = np.abs(unit.T @ unit)
    np.fill_diagonal(similarity, 0.0)
    count = min(neighbors, train.shape[1] - 1)
    edges = np.zeros_like(similarity)
    if count:
        top = np.argsort(-similarity, axis=1, kind="stable")[:, :count]
        np.put_along_axis(edges, top, np.take_along_axis(similarity, top, axis=1), axis=1)
    edges = np.maximum(edges, edges.T)
    np.fill_diagonal(edges, 1.0)
    inv = 1 / np.sqrt(edges.sum(axis=1))
    return torch.as_tensor(edges * inv[:, None] * inv[None, :], dtype=torch.float32)


class GraphLayer(nn.Module):
    def __init__(self, width, encoder):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.encoder = encoder
        if encoder == "gat":
            self.source = nn.Linear(width, 1, bias=False)
            self.target = nn.Linear(width, 1, bias=False)

    def forward(self, nodes, graph):
        projected = self.projection(nodes)
        if self.encoder == "gat":
            scores = F.leaky_relu(self.source(projected) + self.target(projected).T, 0.2)
            graph = scores.masked_fill(graph == 0, -torch.inf).softmax(dim=1)
        return F.relu(graph @ projected)


class ModalityEncoder(nn.Module):
    def __init__(self, train, columns, hidden, depth, neighbors, encoder):
        super().__init__()
        self.register_buffer("columns", torch.as_tensor(columns, dtype=torch.long))
        self.register_buffer("graph", feature_graph(train[:, columns], neighbors))
        self.node_features = nn.Parameter(torch.empty(len(columns), hidden))
        nn.init.xavier_uniform_(self.node_features)
        self.layers = nn.ModuleList(GraphLayer(hidden, encoder) for _ in range(depth))
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden, 2))

    def forward(self, x):
        nodes = self.node_features
        for layer in self.layers:
            nodes = F.dropout(layer(nodes, self.graph), 0.15, self.training)
        # Feature-mask selection is already applied to columns and graph vertices.
        readout = x[:, self.columns] @ nodes / len(self.columns)
        return self.head(readout).log_softmax(dim=1)


class Ensemble(nn.Module):
    def __init__(self, train, groups, mask, hidden, depth, neighbors, encoder):
        super().__init__()
        self.branches = nn.ModuleList(
            ModalityEncoder(train, cols[mask[cols]], hidden, depth, neighbors, encoder)
            for cols in groups
        )
        self.fusion_logits = nn.Parameter(torch.zeros(len(groups)))

    def forward(self, x):
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        log_weights = self.fusion_logits.log_softmax(dim=0)
        return torch.logsumexp(branches + log_weights[None, :, None], dim=1)


def fit_candidate(
    train,
    labels,
    validation,
    val_labels,
    groups,
    config,
    neighbors,
    encoder,
    epochs,
    patience,
    seed,
):
    torch.manual_seed(seed)
    model = Ensemble(train, groups, config.mask, config.hidden, config.depth, neighbors, encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    x, y = torch.from_numpy(train), torch.from_numpy(labels)
    xv, yv = torch.from_numpy(validation), torch.from_numpy(val_labels)
    best_loss, stale = float("inf"), 0
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(x), y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.nll_loss(model(xv), yv).item()
        if val_loss < best_loss:
            best_state, best_loss, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("Training produced a non-finite validation loss. Check feature ranges.")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return model(torch.from_numpy(np.asarray(x, dtype=np.float32))).exp().numpy()
