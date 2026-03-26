from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class SemanticGraphBundle:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_entities: int
    num_relations: int
    pad_entity_id: int
    item_ids: list[int] = field(default_factory=list)
    text_feature_matrix: torch.Tensor | None = None
    visual_feature_matrix: torch.Tensor | None = None


@dataclass(slots=True)
class ConversationTurnState:
    current_turn: torch.Tensor
    previous_turn: torch.Tensor | None = None
    routing_momentum: torch.Tensor | None = None


@dataclass(slots=True)
class DCMoMEBatch:
    context_input_ids: torch.Tensor
    context_attention_mask: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    entity_ids: torch.Tensor
    entity_mask: torch.Tensor
    current_turn_input_ids: torch.Tensor
    current_turn_attention_mask: torch.Tensor
    previous_turn_input_ids: torch.Tensor | None = None
    previous_turn_attention_mask: torch.Tensor | None = None
    turn_history_input_ids: torch.Tensor | None = None
    turn_history_attention_mask: torch.Tensor | None = None
    turn_history_mask: torch.Tensor | None = None
    text_features: torch.Tensor | None = None
    visual_features: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    rec_labels: torch.Tensor | None = None
    conversation_input_ids: torch.Tensor | None = None
    conversation_attention_mask: torch.Tensor | None = None
    conversation_labels: torch.Tensor | None = None
    retrieved_prompt_input_ids: torch.Tensor | None = None
    retrieved_prompt_attention_mask: torch.Tensor | None = None
    generation_input_ids: torch.Tensor | None = None
    generation_attention_mask: torch.Tensor | None = None
    response_input_ids: torch.Tensor | None = None


@dataclass(slots=True)
class DCMoMEForwardOutput:
    prompt_embeds: torch.Tensor
    prompt_token_embeds: torch.Tensor
    fused_entity_embeds: torch.Tensor
    h_kg: torch.Tensor
    h_t: torch.Tensor
    h_v: torch.Tensor
    expert_h_kg: torch.Tensor
    expert_h_t: torch.Tensor
    expert_h_v: torch.Tensor
    routing_weights: torch.Tensor
    current_routing_weights: torch.Tensor
    routing_momentum: torch.Tensor
    topic_shift: torch.Tensor | None
    projected_modalities: dict[str, torch.Tensor]
    recommendation_logits: torch.Tensor | None = None
    recommendation_candidate_ids: torch.Tensor | None = None
