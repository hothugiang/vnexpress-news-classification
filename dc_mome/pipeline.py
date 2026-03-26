from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import DCMoMEConfig
from .encoders import DialogueTurnEncoder, ModalityProjector, RelationAwareKGEncoder
from .model_prompt import DCMoMEKGPrompt
from .router import DialogueConditionedRouter, ModalityExpert, MomentumDriftController
from .structures import DCMoMEBatch, DCMoMEForwardOutput, SemanticGraphBundle


class DCMoMEModel(nn.Module):
    def __init__(
        self,
        config: DCMoMEConfig,
        graph_bundle: SemanticGraphBundle,
        dialogue_backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.prompt.hidden_size
        self.kg_encoder = RelationAwareKGEncoder(
            num_entities=graph_bundle.num_entities,
            num_relations=graph_bundle.num_relations,
            hidden_dim=config.encoder.kg_dim,
            num_bases=config.encoder.num_bases,
            edge_index=graph_bundle.edge_index,
            edge_type=graph_bundle.edge_type,
        )
        self.dialogue_encoder = DialogueTurnEncoder(
            dialogue_backbone, config.encoder.dialogue_dim
        )
        self.prompt_text_backbone = dialogue_backbone
        self.projectors = nn.ModuleDict(
            {
                "kg": ModalityProjector(
                    config.encoder.kg_dim,
                    config.encoder.projector_hidden_dim,
                    hidden_size,
                ),
                "text": ModalityProjector(
                    config.encoder.text_dim,
                    config.encoder.projector_hidden_dim,
                    hidden_size,
                ),
                "visual": ModalityProjector(
                    config.encoder.visual_dim,
                    config.encoder.projector_hidden_dim,
                    hidden_size,
                ),
            }
        )
        self.experts = nn.ModuleDict(
            {
                name: ModalityExpert(hidden_size, config.encoder.expert_hidden_dim)
                for name in ("kg", "text", "visual")
            }
        )
        self.router = DialogueConditionedRouter(
            dialogue_dim=config.encoder.dialogue_dim,
            hidden_size=hidden_size,
            key_dim=config.encoder.router_key_dim,
        )
        self.momentum = MomentumDriftController(
            dialogue_dim=config.encoder.dialogue_dim,
            beta=config.encoder.beta,
        )
        self.prompt_formatter = DCMoMEKGPrompt(
            hidden_size=hidden_size,
            token_hidden_size=config.encoder.dialogue_dim,
            n_head=config.prompt.n_head,
            n_layer=config.prompt.n_layer,
            n_block=config.prompt.n_block,
            n_prefix_rec=config.prompt.n_prefix_rec,
            n_prefix_conv=config.prompt.n_prefix_conv,
        )
        self.register_buffer(
            "item_ids",
            torch.as_tensor(graph_bundle.item_ids, dtype=torch.long)
            if graph_bundle.item_ids
            else torch.zeros(0, dtype=torch.long),
        )
        if graph_bundle.text_feature_matrix is None:
            graph_bundle.text_feature_matrix = torch.zeros(
                (graph_bundle.num_entities, config.encoder.text_dim),
                dtype=torch.float32,
            )
        if graph_bundle.visual_feature_matrix is None:
            graph_bundle.visual_feature_matrix = torch.zeros(
                (graph_bundle.num_entities, config.encoder.visual_dim),
                dtype=torch.float32,
            )
        self.register_buffer(
            "global_text_features", graph_bundle.text_feature_matrix.float()
        )
        self.register_buffer(
            "global_visual_features", graph_bundle.visual_feature_matrix.float()
        )
        self.global_router_logits = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        batch: DCMoMEBatch,
        previous_momentum: torch.Tensor | None = None,
        use_rec_prefix: bool = False,
        use_conv_prefix: bool = False,
        prompt_output_entity: bool = True,
    ) -> DCMoMEForwardOutput:
        kg_states = self.kg_encoder()[batch.entity_ids]
        h_kg = self.projectors["kg"](kg_states)
        text_features, text_presence = self._lookup_entity_modality_features(
            batch.entity_ids, "text"
        )
        visual_features, visual_presence = self._lookup_entity_modality_features(
            batch.entity_ids, "visual"
        )
        h_t = self.projectors["text"](text_features)
        h_v = self.projectors["visual"](visual_features)
        projected = {"kg": h_kg, "text": h_t, "visual": h_v}
        expert_h_kg = self.experts["kg"](h_kg)
        expert_h_t = self.experts["text"](h_t)
        expert_h_v = self.experts["visual"](h_v)
        expert_states = {"kg": expert_h_kg, "text": expert_h_t, "visual": expert_h_v}

        modality_mask = self._build_modality_mask(batch, text_presence, visual_presence)
        current_turn, previous_turn, current_routing, history_momentum, has_history = (
            self._compute_turn_aware_routing_inputs(
                batch=batch,
                projected=projected,
                modality_mask=modality_mask,
            )
        )
        if history_momentum is not None:
            previous_momentum = history_momentum
        final_routing, routing_momentum, topic_shift = self.momentum(
            current_dialogue_turn=current_turn,
            current_routing_weights=current_routing,
            previous_dialogue_turn=previous_turn,
            previous_momentum=previous_momentum,
            history_mask=has_history,
        )

        fused_entity_embeds = sum(
            final_routing[:, idx : idx + 1, :].transpose(1, 2) * expert_states[name]
            for idx, name in enumerate(("kg", "text", "visual"))
        )
        prompt_token_states = self.encode_prompt_tokens(
            batch.prompt_input_ids,
            batch.prompt_attention_mask,
        )
        if (
            not prompt_output_entity
            and batch.retrieved_prompt_input_ids is not None
            and batch.retrieved_prompt_attention_mask is not None
        ):
            prompt_token_states = self.encode_retrieved_prompt_tokens(
                batch.retrieved_prompt_input_ids,
                batch.retrieved_prompt_attention_mask,
                batch.entity_ids.size(0),
                self.config.data.n_examples,
                self.config.data.prompt_max_length,
            )
        if prompt_output_entity:
            prompt_token_embeds, prompt_embeds = self.prompt_formatter.build_prompt_kv(
                entity_embeds=fused_entity_embeds,
                token_embeds=prompt_token_states,
                output_entity=True,
                use_rec_prefix=use_rec_prefix,
                use_conv_prefix=False,
            )
        else:
            prompt_token_embeds = self.prompt_formatter.build_conv_prompt_tokens(
                entity_embeds=fused_entity_embeds,
                token_embeds=prompt_token_states,
                use_conv_prefix=use_conv_prefix,
            )
            prompt_embeds = prompt_token_embeds.new_zeros(
                (
                    self.config.prompt.n_layer,
                    self.config.prompt.n_block,
                    prompt_token_embeds.size(0),
                    self.config.prompt.n_head,
                    prompt_token_embeds.size(1),
                    self.config.prompt.hidden_size // self.config.prompt.n_head,
                )
            )
        return DCMoMEForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_token_embeds=prompt_token_embeds,
            fused_entity_embeds=fused_entity_embeds,
            h_kg=h_kg,
            h_t=h_t,
            h_v=h_v,
            expert_h_kg=expert_h_kg,
            expert_h_t=expert_h_t,
            expert_h_v=expert_h_v,
            routing_weights=final_routing,
            current_routing_weights=current_routing,
            routing_momentum=routing_momentum,
            topic_shift=topic_shift,
            projected_modalities=projected,
        )

    def _compute_turn_aware_routing_inputs(
        self,
        batch: DCMoMEBatch,
        projected: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if (
            batch.turn_history_input_ids is None
            or batch.turn_history_attention_mask is None
            or batch.turn_history_mask is None
        ):
            current_turn = self.dialogue_encoder(
                batch.current_turn_input_ids,
                batch.current_turn_attention_mask,
            )
            previous_turn = None
            if (
                batch.previous_turn_input_ids is not None
                and batch.previous_turn_attention_mask is not None
            ):
                previous_turn = self.dialogue_encoder(
                    batch.previous_turn_input_ids,
                    batch.previous_turn_attention_mask,
                )
            current_routing = self.router(
                current_turn, projected, modality_mask=modality_mask
            )
            return current_turn, previous_turn, current_routing, None, None

        turn_states = self._encode_turn_history(
            batch.turn_history_input_ids, batch.turn_history_attention_mask
        )
        turn_mask = batch.turn_history_mask.bool()
        routing_history = []
        for turn_idx in range(turn_states.size(1)):
            routing_history.append(
                self.router(
                    turn_states[:, turn_idx, :], projected, modality_mask=modality_mask
                )
            )
        routing_history = torch.stack(routing_history, dim=1)

        current_turn, current_routing = self._gather_last_turn(
            turn_states, routing_history, turn_mask
        )
        previous_turn = self._gather_previous_turn(turn_states, turn_mask)
        history_momentum, has_history = self._compute_history_momentum(
            routing_history, turn_mask, modality_mask
        )
        return (
            current_turn,
            previous_turn,
            current_routing,
            history_momentum,
            has_history,
        )

    def _encode_turn_history(
        self,
        turn_history_input_ids: torch.Tensor,
        turn_history_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_turns, seq_len = turn_history_input_ids.shape
        flat_states = self.dialogue_encoder(
            turn_history_input_ids.reshape(batch_size * num_turns, seq_len),
            turn_history_attention_mask.reshape(batch_size * num_turns, seq_len),
        )
        return flat_states.view(batch_size, num_turns, -1)

    @staticmethod
    def _gather_last_turn(
        turn_states: torch.Tensor,
        routing_history: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        turn_counts = turn_mask.long().sum(dim=1).clamp(min=1)
        last_idx = turn_counts - 1
        batch_idx = torch.arange(turn_states.size(0), device=turn_states.device)
        return turn_states[batch_idx, last_idx], routing_history[batch_idx, last_idx]

    @staticmethod
    def _gather_previous_turn(
        turn_states: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        turn_counts = turn_mask.long().sum(dim=1)
        if (turn_counts > 1).sum() == 0:
            return None
        previous_idx = (turn_counts - 2).clamp(min=0)
        batch_idx = torch.arange(turn_states.size(0), device=turn_states.device)
        previous_turn = turn_states[batch_idx, previous_idx]
        fallback_turn = turn_states[batch_idx, (turn_counts - 1).clamp(min=0)]
        has_previous = (turn_counts > 1).view(-1, 1)
        return torch.where(has_previous, previous_turn, fallback_turn)

    def _compute_history_momentum(
        self,
        routing_history: torch.Tensor,
        turn_mask: torch.Tensor,
        modality_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch_size = routing_history.size(0)
        has_history = turn_mask.long().sum(dim=1) > 1
        if has_history.sum() == 0:
            return None, has_history

        modality_available = torch.stack(
            [modality_mask[name].bool() for name in ("kg", "text", "visual")],
            dim=1,
        )
        init = modality_available.float()
        denom = init.sum(dim=1, keepdim=True).clamp(min=1.0)
        init = init / denom
        momentum = init.clone()

        for batch_idx in range(batch_size):
            if not has_history[batch_idx]:
                continue
            valid_turns = torch.nonzero(turn_mask[batch_idx], as_tuple=False).flatten()
            for turn_idx in valid_turns[:-1]:
                momentum[batch_idx] = (
                    self.momentum.beta * momentum[batch_idx]
                    + (1.0 - self.momentum.beta)
                    * routing_history[batch_idx, int(turn_idx)]
                )
        momentum = momentum.masked_fill(~modality_available, 0.0)
        momentum = momentum / momentum.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return momentum, has_history

    def _lookup_entity_modality_features(
        self,
        entity_ids: torch.Tensor,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality == "text":
            feature_bank = self.global_text_features
        elif modality == "visual":
            feature_bank = self.global_visual_features
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        features = feature_bank[entity_ids]
        presence = features.abs().sum(dim=-1) > 0
        return features, presence

    @staticmethod
    def _build_modality_mask(
        batch: DCMoMEBatch,
        text_presence: torch.Tensor,
        visual_presence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        entity_mask = batch.entity_mask
        masks = {"kg": entity_mask}
        masks["text"] = entity_mask * text_presence.long()
        masks["visual"] = entity_mask * visual_presence.long()
        return masks

    def lookup_modality_presence(
        self, entity_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_presence = self.global_text_features[entity_ids].abs().sum(dim=-1) > 0
        visual_presence = self.global_visual_features[entity_ids].abs().sum(dim=-1) > 0
        return text_presence.float(), visual_presence.float()

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")

    def load(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        state_dict = torch.load(input_dir / "model.pt", map_location="cpu")
        self.load_state_dict(state_dict)

    def get_all_entity_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_kg = self.kg_encoder()
        entity_ids = torch.arange(
            all_kg.size(0), device=all_kg.device, dtype=torch.long
        )
        h_kg = self.projectors["kg"](all_kg)
        h_t = self.projectors["text"](self.global_text_features.to(all_kg.device))
        h_v = self.projectors["visual"](self.global_visual_features.to(all_kg.device))
        expert_h_kg = self.experts["kg"](h_kg)
        expert_h_t = self.experts["text"](h_t)
        expert_h_v = self.experts["visual"](h_v)
        masks = torch.stack(
            [
                torch.ones(all_kg.size(0), device=all_kg.device, dtype=torch.bool),
                self.global_text_features.to(all_kg.device).abs().sum(dim=-1) > 0,
                self.global_visual_features.to(all_kg.device).abs().sum(dim=-1) > 0,
            ],
            dim=-1,
        )
        logits = self.global_router_logits.view(1, 3).expand(all_kg.size(0), -1)
        logits = logits.masked_fill(~masks, -1e4)
        weights = torch.softmax(logits, dim=-1)
        fused = (
            weights[:, 0:1] * expert_h_kg
            + weights[:, 1:2] * expert_h_t
            + weights[:, 2:3] * expert_h_v
        )
        return fused, entity_ids

    def encode_prompt_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.prompt_text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def encode_retrieved_prompt_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        n_examples: int,
        prompt_max_length: int,
    ) -> torch.Tensor:
        hidden_states = self.encode_prompt_tokens(input_ids, attention_mask)
        hidden_states = hidden_states[:, :prompt_max_length, :]
        hidden_states = hidden_states.contiguous().view(
            batch_size, n_examples * hidden_states.size(1), hidden_states.size(2)
        )
        return hidden_states

    def forward_entity_summary(self, entity_ids: torch.Tensor) -> torch.Tensor:
        all_entity_embeds, _ = self.get_all_entity_embeds()
        return all_entity_embeds[entity_ids]
