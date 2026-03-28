import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv


class RoutingState:
    def __init__(self, batch_size, entity_len, device=None):
        self.prev_g_hat = None  # g_hat^{(b-1)}
        self.prev_momentum = None  # g_mom^{(b-1)}
        self.prev_dialogue_emb = None
        self.current_turn = 0

    def reset(self):
        self.prev_g_hat = None
        self.prev_momentum = None
        self.prev_dialogue_emb = None
        self.current_turn = 0


class DCMoMEPrompt(nn.Module):
    def __init__(
        self,
        hidden_size,
        token_hidden_size,
        n_head,
        n_layer,
        n_block,
        n_entity,
        num_relations,
        num_bases,
        edge_index,
        edge_type,
        text_embeddings,
        image_embeddings,
        id_to_idx,
        d_ff=512,
        d_k=64,
        n_prefix_rec=None,
        n_prefix_conv=None,
    ):
        super(DCMoMEPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.d_k = d_k
        self.d_ff = d_ff

        self.id_to_idx = id_to_idx
        # self.id_to_idx_tensor = torch.tensor([self.id_to_idx[i] for i in ids])
        id_to_idx_arr = torch.zeros(n_entity, dtype=torch.long)
        for eid, idx in id_to_idx.items():
            if eid < n_entity:
                id_to_idx_arr[eid] = idx
        self.register_buffer("id_to_idx_tensor", id_to_idx_arr)

        self.beta = 0.5
        self.lb_alpha = 0.01

        ## KG Encoder
        entity_hidden_size = hidden_size // 2

        self.kg_encoder = RGCNConv(
            entity_hidden_size,
            entity_hidden_size,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)

        self.register_buffer("text_embeddings", text_embeddings)
        self.register_buffer("image_embeddings", image_embeddings)

        ## Project mỗi modality vào common space ℝ^d
        self.kg_proj = nn.Sequential(
            nn.Linear(entity_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_embeddings.size(1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(image_embeddings.size(1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        ## expert
        self.expert_kg = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.expert_txt = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.expert_vis = nn.Sequential(
            nn.Linear(hidden_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        # ========== 5. Dialogue-Conditioned Gating ==========
        self.gate_query = nn.Linear(hidden_size, hidden_size)  # W_Q
        self.gate_key_kg = nn.Linear(hidden_size, hidden_size)  # W_K^kg
        self.gate_key_txt = nn.Linear(hidden_size, hidden_size)  # W_K^t
        self.gate_key_vis = nn.Linear(hidden_size, hidden_size)  # W_K^v

        # ========== 6. Drift Gate ==========
        drift_input_dim = hidden_size + 1  # [dialogue_emb; topic_shift]
        self.drift_gate = nn.Sequential(
            nn.Linear(drift_input_dim, drift_input_dim // 2),
            nn.ReLU(),
            nn.Linear(drift_input_dim // 2, 1),
            nn.Sigmoid(),
        )

        ## Projectors cho entity và token
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

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

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(
                torch.empty(n_prefix_rec, hidden_size)
            )
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(
                torch.empty(n_prefix_conv, hidden_size)
            )
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        # Bước 1: RGCN + residual  →  e_kg_base  (n_entity, entity_hidden_size)
        node_embeds = self.node_embeds
        e_kg_base = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        e_kg_base = self.entity_proj1(e_kg_base) + e_kg_base  # residual MLP

        # Bước 2: KG expert cho toàn bộ entity
        h_kg_all = self.kg_proj(e_kg_base)  # (n_entity, hidden_size)
        o_kg_all = self.expert_kg(h_kg_all)  # (n_entity, hidden_size)

        # Mặc định: mọi entity đều dùng KG expert thuần
        entity_embeds_all = o_kg_all.clone()

        # Bước 3: Item entities → fuse thêm text + visual (uniform 1/3)
        #         Uniform vì không có dialogue context tại đây
        item_mask = self.id_to_idx_tensor >= 0
        item_kg_ids = torch.where(item_mask)[0]  # index trong KG space
        if item_kg_ids.numel() > 0:
            item_emb_rows = self.id_to_idx_tensor[item_kg_ids]

            h_txt = self.txt_proj(
                self.text_embeddings[item_emb_rows]
            )  # (n_items, hidden)
            h_vis = self.vis_proj(
                self.image_embeddings[item_emb_rows]
            )  # (n_items, hidden)
            o_txt = self.expert_txt(h_txt)
            o_vis = self.expert_vis(h_vis)

            entity_embeds_all[item_kg_ids] = (
                o_kg_all[item_kg_ids] + o_txt + o_vis
            ) / 3.0

        return entity_embeds_all, e_kg_base

    @staticmethod
    def compute_topic_shift(
        curr_emb: torch.Tensor, prev_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Topic shift theo báo cáo §5.3 (Eq. 17):
            δ_t = 1 - cosine_similarity(d_t, d_{t-1})
        curr_emb, prev_emb: [B, hidden_size]
        Trả về: [B] — scalar trong [0, 2], thường ∈ [0, 1]
        """
        # F.cosine_similarity trả về scalar per sample ∈ [-1, 1]
        cos_sim = F.cosine_similarity(curr_emb, prev_emb, dim=-1)  # [B]
        return (1.0 - cos_sim).clamp(min=0.0)  # [B], ∈ [0, 2]

    def compute_routing(
        self,
        h_kg: torch.Tensor,
        h_txt: torch.Tensor,
        h_vis: torch.Tensor,
        dialogue_emb: torch.Tensor,
        state: RoutingState,
    ):
        """
        Tính routing weights g_hat cho mỗi entity.
        Trả về: g_hat (routing weights đã drift-aware), momentum mới, và các weight gốc
        """
        batch_size, entity_len, _ = h_kg.shape
        # [B,1,1,hidden]
        query = self.gate_query(dialogue_emb).unsqueeze(1)

        scores_kg = torch.matmul(
            query, self.gate_key_kg(h_kg).transpose(-2, -1)
        ).squeeze(1) / math.sqrt(self.d_k)
        scores_txt = torch.matmul(
            query, self.gate_key_txt(h_txt).transpose(-2, -1)
        ).squeeze(1) / math.sqrt(self.d_k)
        scores_vis = torch.matmul(
            query, self.gate_key_vis(h_vis).transpose(-2, -1)
        ).squeeze(1) / math.sqrt(self.d_k)
        scores = torch.stack([scores_kg, scores_txt, scores_vis], dim=-1)

        g = F.softmax(scores / math.sqrt(self.hidden_size), dim=-1)  # [B, L, 3]

        # ---- Bước 2: Momentum & Drift ----
        if state.current_turn == 0:
            # Nếu không có lịch sử (turn đầu tiên), dùng g trực tiếp
            g_hat = g
            momentum = g  # g_hat (cũng là momentum ban đầu)
        else:
            # EMA cập nhật momentum
            momentum = (
                self.beta * state.prev_momentum + (1 - self.beta) * state.prev_g_hat
            )

            # Drift gate
            topic_shift = self.compute_topic_shift(
                dialogue_emb, state.prev_dialogue_emb
            )
            topic_shift = topic_shift.view(batch_size, 1, 1)  # [B,1,1]
            drift_input = torch.cat(
                [
                    dialogue_emb.unsqueeze(1).expand(-1, entity_len, -1),
                    topic_shift.expand(-1, entity_len, -1),
                ],
                dim=-1,
            )  # [B, L, hidden+1]
            lambda_b = self.drift_gate(drift_input)  # [B, L, 1]

            # Kết hợp
            g_hat = lambda_b * g + (1 - lambda_b) * momentum

        # Re-normalize trong trường hợp missing modality (đảm bảo tổng = 1)
        g_hat = g_hat / (g_hat.sum(dim=-1, keepdim=True) + 1e-8)
        assert g_hat.dim() == 3, f"g_hat wrong shape: {g_hat.shape}"

        # Update state
        state.prev_g_hat = g_hat.detach()
        state.prev_momentum = momentum.detach()
        state.prev_dialogue_emb = dialogue_emb.detach()
        state.current_turn += 1
        return g_hat, momentum, g

    def forward(
        self,
        entity_ids=None,  # [batch_size, entity_len]
        token_embeds=None,  # [batch_size, token_len, token_hidden_size]
        dialogue_emb=None,  # [B, hidden_size] — turn-level (optional, từ external encoder)
        state=None,
        output_entity=False,
        use_rec_prefix=False,
        use_conv_prefix=False,
    ):
        """
        output_entity: True => trả về prompt_embeds cho recommendation (entity attend to token)
                       False => cho generation (token attend to entity)
        """
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        loss_lb = torch.tensor(0.0, device=self.node_embeds.device)
        loss_cl = torch.tensor(0.0, device=self.node_embeds.device)

        entity_embeds_all, e_kg_base = self.get_entity_embeds()

        if state is None:
            state = RoutingState(batch_size, entity_len)

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            e_kg = e_kg_base[entity_ids]

            # 2b. Project từng modality lên hidden_size
            h_kg = self.kg_proj(e_kg)  # (B, L, hidden)
            real_ids = self.id_to_idx_tensor[entity_ids]  # (B, L)
            h_txt = self.txt_proj(self.text_embeddings[real_ids])  # (B, L, hidden)
            h_vis = self.vis_proj(self.image_embeddings[real_ids])  # (B, L, hidden)

            # 2c. Qua từng expert
            o_kg = self.expert_kg(h_kg)
            o_txt = self.expert_txt(h_txt)
            o_vis = self.expert_vis(h_vis)
            experts = torch.stack([o_kg, o_txt, o_vis], dim=2)  # (B, L, 3, hidden)

            # 2d. Dialogue-conditioned routing + drift gate
            if state is None:
                state = RoutingState()
            d_t = (
                token_embeds.mean(dim=1)
                if token_embeds is not None
                else torch.zeros(
                    batch_size, self.hidden_size, device=self.node_embeds.device
                )
            )
            routing_emb = dialogue_emb if dialogue_emb is not None else d_t

            g_hat, _, g = self.compute_routing(h_kg, h_txt, h_vis, routing_emb, state)
            entity_embeds = (g_hat.unsqueeze(-1) * experts).sum(dim=2)  # (B, L, hidden)

            # 2e. Load Balancing Loss (Switch Transformer)
            P_k = g.mean(dim=[0, 1])
            f_k = (g == g.max(dim=-1, keepdim=True).values).float().mean(dim=[0, 1])
            loss_lb = 3 * (f_k * P_k).sum()
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds
            token_embeds = self.token_proj2(token_embeds)

        # Cross attn
        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(
                0, 2, 1
            )
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                token_weights_embeds = (
                    token_weights @ token_embeds
                )  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(
                    dim=1
                )  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(
                    token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1
                )
                logits /= temperature
                labels = torch.arange(
                    logits.size(0), device=logits.device
                )  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = (
                self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            )
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = (
                self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            )
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size,
            prompt_len,
            self.n_layer,
            self.n_block,
            self.n_head,
            self.head_dim,
        ).permute(2, 3, 0, 4, 1, 5)

        return prompt_embeds, loss_cl, loss_lb, entity_embeds_all

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if "edge" not in k}
        save_path = os.path.join(save_dir, "model.pt")
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, "model.pt")
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device("cpu")), strict=False
        )
        print(missing_keys, unexpected_keys)
