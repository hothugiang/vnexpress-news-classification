from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DCMoMEKGPrompt(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        token_hidden_size: int,
        n_head: int,
        n_layer: int,
        n_block: int,
        n_prefix_rec: int,
        n_prefix_conv: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
        self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
        nn.init.normal_(self.rec_prefix_embeds)
        nn.init.normal_(self.conv_prefix_embeds)
        self.rec_prefix_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.conv_prefix_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def fuse_prompt_tokens(
        self,
        entity_embeds: torch.Tensor | None,
        token_embeds: torch.Tensor | None,
        output_entity: bool,
        use_rec_prefix: bool = False,
        use_conv_prefix: bool = False,
    ) -> torch.Tensor:
        batch_size = None
        entity_len = 0
        token_len = 0

        if entity_embeds is not None:
            batch_size, entity_len = entity_embeds.shape[:2]
        if token_embeds is not None:
            batch_size = token_embeds.shape[0]
            token_len = token_embeds.shape[1]
            token_embeds = self.token_proj1(token_embeds) + token_embeds
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.transpose(1, 2)
            attn_weights = attn_weights / self.hidden_size
            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).transpose(1, 2)
                prompt_tokens = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_tokens = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_tokens = entity_embeds
            prompt_len = entity_len
        elif token_embeds is not None:
            prompt_tokens = token_embeds
            prompt_len = token_len
        else:
            raise ValueError("Either entity_embeds or token_embeds must be provided")

        if use_rec_prefix:
            prefix = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_tokens = torch.cat([prefix, prompt_tokens], dim=1)
            prompt_len += self.n_prefix_rec
        if use_conv_prefix:
            prefix = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_tokens = torch.cat([prefix, prompt_tokens], dim=1)
            prompt_len += self.n_prefix_conv

        if prompt_tokens.size(1) != prompt_len:
            raise RuntimeError("Prompt token length mismatch")
        return prompt_tokens

    def build_prompt_kv(
        self,
        entity_embeds: torch.Tensor | None,
        token_embeds: torch.Tensor | None,
        output_entity: bool,
        use_rec_prefix: bool = False,
        use_conv_prefix: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self.fuse_prompt_tokens(
            entity_embeds=entity_embeds,
            token_embeds=token_embeds,
            output_entity=output_entity,
            use_rec_prefix=use_rec_prefix,
            use_conv_prefix=use_conv_prefix,
        )
        batch_size = prompt_tokens.size(0)
        prompt_tokens = self.prompt_proj1(prompt_tokens) + prompt_tokens
        prompt_kv = self.prompt_proj2(prompt_tokens).reshape(
            batch_size,
            prompt_tokens.size(1),
            self.n_layer,
            self.n_block,
            self.n_head,
            self.head_dim,
        ).permute(2, 3, 0, 4, 1, 5)
        return prompt_tokens, prompt_kv

    def build_conv_prompt_tokens(
        self,
        entity_embeds: torch.Tensor,
        token_embeds: torch.Tensor,
        use_conv_prefix: bool = False,
    ) -> torch.Tensor:
        return self.fuse_prompt_tokens(
            entity_embeds=entity_embeds,
            token_embeds=token_embeds,
            output_entity=False,
            use_conv_prefix=use_conv_prefix,
        )

    def augment_context_for_conversation(
        self,
        prompt_tokens: torch.Tensor,
        entity_embeds: torch.Tensor,
        word_embeddings: torch.Tensor,
        context_input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        mapping: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_prompt = prompt_tokens
        if mapping:
            affinity_scores = self.cross_attn(conv_prompt) @ word_embeddings.transpose(0, 1)
            affinity_scores = affinity_scores / self.hidden_size
            conv_prompt = torch.softmax(affinity_scores, dim=-1) @ word_embeddings
        prompt_attention_mask = torch.ones(
            (conv_prompt.size(0), conv_prompt.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        augmented_embeddings = torch.cat([conv_prompt, context_input_embeddings], dim=1)
        augmented_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        entity_summary = entity_embeds.mean(dim=1, keepdim=True).expand(-1, conv_prompt.size(1), -1)
        return augmented_embeddings, augmented_attention_mask, entity_summary
