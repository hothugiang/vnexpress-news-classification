from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ModalityExpert(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_dim),
            nn.GELU(),
            nn.Linear(expert_hidden_dim, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class DialogueConditionedRouter(nn.Module):
    def __init__(self, dialogue_dim: int, hidden_size: int, key_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dialogue_dim, key_dim)
        self.key_proj = nn.ModuleDict(
            {
                "kg": nn.Linear(hidden_size, key_dim),
                "text": nn.Linear(hidden_size, key_dim),
                "visual": nn.Linear(hidden_size, key_dim),
            }
        )
        self.key_dim = key_dim

    def forward(
        self,
        dialogue_turn: torch.Tensor,
        modality_states: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.query_proj(dialogue_turn).unsqueeze(1)
        scores = []
        for name in ("kg", "text", "visual"):
            key = self.key_proj[name](modality_states[name])
            score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.key_dim)
            scores.append(score)
        logits = torch.cat(scores, dim=1)

        if modality_mask is not None:
            for idx, name in enumerate(("kg", "text", "visual")):
                if name not in modality_mask:
                    continue
                mask = modality_mask[name].unsqueeze(1)
                logits[:, idx : idx + 1, :] = logits[:, idx : idx + 1, :].masked_fill(mask == 0, -1e4)

        return F.softmax(logits, dim=1)


class MomentumDriftController(nn.Module):
    def __init__(self, dialogue_dim: int, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.drift_gate = nn.Linear(dialogue_dim + 1, 1)

    def forward(
        self,
        current_dialogue_turn: torch.Tensor,
        current_routing_weights: torch.Tensor,
        previous_dialogue_turn: torch.Tensor | None = None,
        previous_momentum: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if previous_dialogue_turn is None or previous_momentum is None:
            return current_routing_weights, current_routing_weights, None

        topic_shift = 1.0 - F.cosine_similarity(current_dialogue_turn, previous_dialogue_turn, dim=-1)
        drift_input = torch.cat([current_dialogue_turn, topic_shift.unsqueeze(-1)], dim=-1)
        lambda_b = torch.sigmoid(self.drift_gate(drift_input)).view(-1, 1, 1)
        routed = lambda_b * current_routing_weights + (1.0 - lambda_b) * previous_momentum
        momentum = self.beta * previous_momentum + (1.0 - self.beta) * current_routing_weights
        if history_mask is not None:
            expanded_mask = history_mask.view(-1, 1, 1)
            routed = torch.where(expanded_mask, routed, current_routing_weights)
            momentum = torch.where(expanded_mask, momentum, current_routing_weights)
            topic_shift = torch.where(history_mask, topic_shift, torch.zeros_like(topic_shift))
        return routed, momentum, topic_shift
