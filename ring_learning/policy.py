"""Compact state-based behavior-cloning policy, with portable checkpoints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Normalizer


class Policy(nn.Module):
    def __init__(self, obs_size: int = 30, hidden_size: int = 128, feature_indices=None):
        super().__init__()
        self.obs_size = int(obs_size)
        self.hidden_size = int(hidden_size)
        self.feature_indices = list(range(obs_size)) if feature_indices is None else list(feature_indices)
        if not self.feature_indices or any(i < 0 or i >= obs_size for i in self.feature_indices):
            raise ValueError("Feature indices must refer to observation columns")
        self.net = nn.Sequential(nn.Linear(len(self.feature_indices), hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, 4), nn.Tanh())

    def forward(self, normalized_observation):
        return self.net(normalized_observation[..., self.feature_indices])

    def act(self, observation, normalizer: Normalizer):
        self.eval()
        values = normalizer.transform(observation)
        with torch.inference_mode():
            return self(torch.as_tensor(values, dtype=torch.float32)).cpu().numpy()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def save_checkpoint(path, model: Policy, normalizer: Normalizer, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_size": model.obs_size,
                "hidden_size": model.hidden_size, "feature_indices": model.feature_indices,
                "normalizer": normalizer.as_dict(), "metadata": metadata}, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = Policy(payload["obs_size"], payload["hidden_size"], payload["feature_indices"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, Normalizer.from_dict(payload["normalizer"]), payload["metadata"]
