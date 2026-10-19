from __future__ import annotations

import torch
from torch import nn

OBSERVATION_SIZE = 56
ACTION_SIZE = 13


class ActorCritic(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, ACTION_SIZE)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def distribution(self, observation: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, value = self(observation)
        masked_logits = logits.masked_fill(legal_mask <= 0, -1e9)
        return torch.distributions.Categorical(logits=masked_logits), value

    @torch.no_grad()
    def choose(self, observation: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        distribution, _ = self.distribution(observation, legal_mask)
        return distribution.sample()

