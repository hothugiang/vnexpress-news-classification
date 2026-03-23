from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    valid_mask = mask.reshape(-1).bool()
    if valid_mask.sum() <= 1:
        return anchor.new_zeros(())
    anchor = F.normalize(anchor.reshape(-1, anchor.size(-1))[valid_mask], p=2, dim=-1)
    positive = F.normalize(positive.reshape(-1, positive.size(-1))[valid_mask], p=2, dim=-1)
    logits = (anchor @ positive.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def multimodal_alignment_loss(
    h_kg: torch.Tensor,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    kg_mask: torch.Tensor,
    text_mask: torch.Tensor,
    visual_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses = []
    pair_masks = (
        (h_kg, h_t, kg_mask * text_mask),
        (h_kg, h_v, kg_mask * visual_mask),
        (h_t, h_v, text_mask * visual_mask),
    )
    for lhs, rhs, pair_mask in pair_masks:
        if pair_mask.sum() > 1:
            losses.append(masked_info_nce_loss(lhs, rhs, pair_mask, temperature))
    if not losses:
        return h_kg.new_zeros(())
    return sum(losses) / len(losses)


def load_balancing_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    num_experts = routing_weights.size(1)
    mean_prob = routing_weights.mean(dim=(0, 2))
    dominant = routing_weights.argmax(dim=1)
    fraction = torch.stack(
        [(dominant == expert_id).float().mean() for expert_id in range(num_experts)],
        dim=0,
    )
    return num_experts * (fraction * mean_prob).sum()


def recommendation_ce_loss(
    recommendation_logits: torch.Tensor | None,
    rec_labels: torch.Tensor | None,
) -> torch.Tensor:
    if recommendation_logits is None or rec_labels is None:
        ref = recommendation_logits if recommendation_logits is not None else rec_labels
        if ref is None:
            return torch.zeros(())
        return ref.new_zeros(())
    valid_mask = (rec_labels >= 0).float()
    if valid_mask.sum() == 0:
        return recommendation_logits.new_zeros(())
    per_sample = F.cross_entropy(recommendation_logits, rec_labels, reduction="none")
    return (per_sample * valid_mask).sum() / valid_mask.sum()


def generation_loss(lm_loss: torch.Tensor | None) -> torch.Tensor:
    if lm_loss is None:
        return torch.zeros(())
    return lm_loss
