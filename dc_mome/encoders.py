from __future__ import annotations

import math

import torch
from torch import nn

try:
    from torch_geometric.nn import RGCNConv
except ImportError:  # pragma: no cover - optional dependency in the workspace
    RGCNConv = None


class RelationAwareKGEncoder(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int,
        num_bases: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> None:
        super().__init__()
        if RGCNConv is None:
            raise ImportError("torch_geometric is required for RelationAwareKGEncoder")
        self.node_embeds = nn.Parameter(torch.empty(num_entities, hidden_dim))
        stdv = math.sqrt(6.0 / (num_entities + hidden_dim))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.rgcn = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

    def forward(self) -> torch.Tensor:
        return self.rgcn(self.node_embeds, self.edge_index, self.edge_type) + self.node_embeds


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DialogueTurnEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.attn_pool = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        weights = self.attn_pool(hidden_states).squeeze(-1)
        weights = weights.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1)
        return torch.einsum("bs,bsh->bh", weights, hidden_states)
