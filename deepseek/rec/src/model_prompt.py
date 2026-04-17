import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class RoutingState:
    def __init__(self, batch_size=None, entity_len=None, device=None):
        self.prev_g_hat = None  # g_hat^{(b-1)}
        self.prev_momentum = None  # g_mom^{(b-1)}
        self.prev_dialogue_emb = None
        self.current_turn = 0

    def reset(self):
        self.prev_g_hat = None
        self.prev_momentum = None
        self.prev_dialogue_emb = None
        self.current_turn = 0


class MoMELayer(nn.Module):
    """
    Mixture of Modality Experts.

    Với mỗi entity:
      - Nếu là movie  → gating(kg_emb) cho ra [w_kg, w_txt, w_img], softmax
      - Nếu không phải movie → chỉ dùng kg_emb (w_kg = 1)

    Output: fused embedding cùng chiều với entity_hidden_size
    """

    def __init__(
        self,
        entity_hidden_size: int,
        text_hidden_size: int,
        image_hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_hidden_size = entity_hidden_size

        # Project text và image về entity_hidden_size nếu khác chiều
        self.text_proj = nn.Linear(text_hidden_size, entity_hidden_size)
        self.image_proj = nn.Linear(image_hidden_size, entity_hidden_size)

        # Gating network: nhận kg_emb, cho ra 3 logit
        self.gate = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # thêm
            nn.Linear(entity_hidden_size // 2, 3),  # [w_kg, w_txt, w_img]
        )

    def forward(
        self,
        kg_embeds: torch.Tensor,  # (N, entity_hidden_size)  — tất cả entity
        text_embeddings: torch.Tensor,  # (M, text_hidden_size)    — chỉ movie (M ≤ N)
        image_embeddings: torch.Tensor,  # (M, image_hidden_size)   — chỉ movie
        movie_indices: torch.Tensor,  # (M,) long — vị trí của movie trong N entity
    ) -> torch.Tensor:
        N = kg_embeds.size(0)
        device = kg_embeds.device

        # Project text/image → entity space
        txt_proj = self.text_proj(text_embeddings)  # (M, D)
        img_proj = self.image_proj(image_embeddings)  # (M, D)

        # Tính gating weights chỉ cho movie
        kg_movie = kg_embeds[movie_indices]  # (M, D)
        logits = self.gate(kg_movie)  # (M, 3)
        weights = F.softmax(logits, dim=-1)  # (M, 3)  [w_kg, w_txt, w_img]
        # print(
        #     f"Gate weights mean - kg:{weights[:, 0].mean():.3f}, "
        #     f"txt:{weights[:, 1].mean():.3f}, "
        #     f"img:{weights[:, 2].mean():.3f}"
        # )
        # print(
        #     f"Gate weights std  - kg:{weights[:, 0].std():.3f}, "
        #     f"txt:{weights[:, 1].std():.3f}, "
        #     f"img:{weights[:, 2].std():.3f}"
        # )
        w_kg = weights[:, 0:1]  # (M, 1)
        w_txt = weights[:, 1:2]
        w_img = weights[:, 2:3]

        # Fused embedding cho movie
        alpha = 0.2  # text/image luôn contribute ít nhất 20%/2 = 10% mỗi cái
        fused_movie = (1 - alpha) * (
            w_kg * kg_movie + w_txt * txt_proj + w_img * img_proj
        ) + alpha * (txt_proj + img_proj) / 2
        # Bắt đầu từ kg_embeds (non-movie giữ nguyên)
        fused = kg_embeds.clone()
        fused[movie_indices] = fused_movie

        return fused


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
        edge_index_c,
        text_embeddings,
        image_embeddings,
        id_to_idx,
        movie_entity_ids,
        d_ff=512,
        d_k=64,
        n_prefix_rec=None,
        n_prefix_conv=None,
        dropout=0.3,
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

        # self.id_to_idx = id_to_idx
        # # self.id_to_idx_tensor = torch.tensor([self.id_to_idx[i] for i in ids])
        # id_to_idx_arr = torch.zeros(n_entity, dtype=torch.long)
        # for eid, idx in id_to_idx.items():
        #     if eid < n_entity:
        #         id_to_idx_arr[eid] = idx
        # self.register_buffer("id_to_idx_tensor", id_to_idx_arr)

        self.beta = 0.5
        self.lb_alpha = 0.01

        ## KG Encoder
        entity_hidden_size = hidden_size // 2
        _text_hidden_size = text_embeddings.size(-1)
        _image_hidden_size = image_embeddings.size(-1)

        self.kg_encoder = RGCNConv(
            entity_hidden_size,
            entity_hidden_size,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.conv_c1 = CustomGCNConv()
        self.conv_c2 = CustomGCNConv()
        self.conv_c3 = CustomGCNConv()
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.register_buffer("edge_index_c", edge_index_c)

        self.register_buffer("text_embeddings", text_embeddings)
        self.register_buffer("image_embeddings", image_embeddings)

        ## Project mỗi modality vào common space ℝ^d
        self.mome = MoMELayer(
            entity_hidden_size, _text_hidden_size, _image_hidden_size, dropout=dropout
        )
        if not isinstance(movie_entity_ids, torch.Tensor):
            movie_entity_ids = torch.tensor(movie_entity_ids, dtype=torch.long)
        self.register_buffer("movie_entity_ids", movie_entity_ids)
        ## expert
        # self.expert_kg = nn.Sequential(
        #     nn.Linear(hidden_size, d_ff),
        #     nn.GELU(),
        #     nn.Linear(d_ff, hidden_size),
        #     nn.LayerNorm(hidden_size),
        # )
        # self.expert_txt = nn.Sequential(
        #     nn.Linear(hidden_size, d_ff),
        #     nn.GELU(),
        #     nn.Linear(d_ff, hidden_size),
        #     nn.LayerNorm(hidden_size),
        # )
        # self.expert_vis = nn.Sequential(
        #     nn.Linear(hidden_size, d_ff),
        #     nn.GELU(),
        #     nn.Linear(d_ff, hidden_size),
        #     nn.LayerNorm(hidden_size),
        # )
        # ========== 5. Dialogue-Conditioned Gating ==========
        self.gate_query = nn.Linear(hidden_size, hidden_size)  # W_Q
        self.gate_key_kg = nn.Linear(hidden_size, hidden_size)  # W_K^kg
        self.gate_key_txt = nn.Linear(hidden_size, hidden_size)  # W_K^t
        self.gate_key_vis = nn.Linear(hidden_size, hidden_size)  # W_K^v

        # ========== 6. Drift Gate ==========
        # drift_input_dim = hidden_size + 1  # [dialogue_emb; topic_shift]
        # self.drift_gate = nn.Sequential(
        #     nn.Linear(drift_input_dim, drift_input_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(drift_input_dim // 2, 1),
        #     nn.Sigmoid(),
        # )

        ## Projectors cho entity và token
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # thêm
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # thêm
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # thêm
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
        entity_embeds = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        # entity_embeds = self.entity_proj2(entity_embeds)
        # sorted_indices = self.sorted_indices.to(entity_embeds.device)

        entity_embeds = self.mome(
            kg_embeds=entity_embeds,
            text_embeddings=self.text_embeddings,
            image_embeddings=self.image_embeddings,
            movie_indices=self.movie_entity_ids,
        )
        ec1 = self.conv_c1(entity_embeds, self.edge_index_c)
        ec2 = self.conv_c2(ec1, self.edge_index_c)
        ec3 = self.conv_c3(ec2, self.edge_index_c)
        entity_embeds = (entity_embeds + ec1 + ec2 + ec3) / 4  # (N, D)

        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds

        self.loss_align = torch.tensor(0.0, device=entity_embeds.device)
        kg_movie = entity_embeds[self.movie_entity_ids].detach()  # (M, 384)
        txt_proj = self.mome.text_proj(self.text_embeddings)  # (M, 384)
        img_proj = self.mome.image_proj(self.image_embeddings)  # (M, 384)

        # Normalize
        kg_n = F.normalize(kg_movie, dim=-1)
        txt_n = F.normalize(txt_proj, dim=-1)
        img_n = F.normalize(img_proj, dim=-1)

        # Contrastive: movie i là positive pair của chính nó across modalities
        # txt↔kg
        logits_tk = txt_n @ kg_n.T / 0.07  # (M, M)
        labels = torch.arange(logits_tk.size(0), device=entity_embeds.device)
        loss_tk = F.cross_entropy(logits_tk, labels)

        # img↔kg
        logits_ik = img_n @ kg_n.T / 0.07
        loss_ik = F.cross_entropy(logits_ik, labels)

        # txt↔img
        logits_ti = txt_n @ img_n.T / 0.07
        loss_ti = F.cross_entropy(logits_ti, labels)

        self.loss_align = (loss_tk + loss_ik + loss_ti) / 3

        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

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
        modality_mask=None,
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

        g = F.softmax(scores / (math.sqrt(self.d_k) * 0.5), dim=-1)  # [B, L, 3]
        return g, g, g
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

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]

        if state is None:
            state = RoutingState(batch_size, entity_len)

        # if entity_ids is not None:
        #     batch_size, entity_len = entity_ids.shape[:2]
        #     e_kg = entity_embeds.copy()

        #     # 2b. Project từng modality lên hidden_size
        #     h_kg = self.kg_proj(e_kg)  # (B, L, hidden)
        #     real_ids = self.id_to_idx_tensor[entity_ids]  # (B, L)
        #     h_txt = self.txt_proj(self.text_embeddings[real_ids])  # (B, L, hidden)
        #     h_vis = self.vis_proj(self.image_embeddings[real_ids])  # (B, L, hidden)

        #     # 2c. Qua từng expert
        #     o_kg = self.expert_kg(h_kg)
        #     o_txt = self.expert_txt(h_txt)
        #     o_vis = self.expert_vis(h_vis)
        #     experts = torch.stack([o_kg, o_txt, o_vis], dim=2)  # (B, L, 3, hidden)

        #     # 2d. Dialogue-conditioned routing + drift gate
        #     if state is None:
        #         state = RoutingState()
        #     d_t = (
        #         token_embeds[:, 0, :]
        #     if token_embeds is not None
        #     else torch.zeros(
        #         batch_size, self.hidden_size, device=self.node_embeds.device
        #     )
        # )
        # routing_emb = dialogue_emb if dialogue_emb is not None else d_t

        # g_hat, _, g = self.compute_routing(h_kg, h_txt, h_vis, routing_emb, state)
        # entity_embeds = (g_hat.unsqueeze(-1) * experts).sum(dim=2)  # (B, L, hidden)

        # # 2e. Load Balancing Loss (Switch Transformer)
        # P_k = g.mean(dim=[0, 1])
        # f_k = (g == g.max(dim=-1, keepdim=True).values).float().mean(dim=[0, 1])
        # loss_lb = 3 * (f_k * P_k).sum()
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

        return prompt_embeds, loss_cl

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


"""
UH-CSG Prompt Encoder
=====================
Replaces MSCRS's 3 parallel GCN pipelines with:
  1. R-GCN on DBpedia KG (same as MSCRS)
  2. LightGCN on Unified Heterogeneous CSG
  3. Gated fusion of content vs collaborative signals

Drop-in replacement: output shape and interface identical to MMPrompt.
"""

import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from loguru import logger


class CustomGCNConv(MessagePassing):
    """LightGCN-style convolution: no learnable weights, symmetric normalization."""

    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class UHCSGPrompt(nn.Module):
    """
    UH-CSG Prompt Encoder.

    Changes vs MMPrompt:
      - Removes: conv_c1/c2/c3, conv_ts1/ts2/ts3, conv_is1/is2/is3
      - Adds: unified LightGCN on CSG + gated fusion
      - Keeps: R-GCN (kg_encoder), cross_attn, prompt projections, prefix embeddings
    """

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
        edge_index,  # KG edges
        edge_type,  # KG edge types
        # --- UH-CSG specific ---
        unified_edge_index,  # [2, E] unified CSG edges (global node space)
        num_dialogues,  # M: number of dialogue session nodes
        num_ml_users,  # K: number of MovieLens user nodes
        dialogue_item_map,  # dict: dialogue_idx -> list of entity IDs
        movielens_edges=None,  # [2, E_ml] for user node init (user_local_idx, entity_id)
        # --- Standard params ---
        num_lightgcn_layers=4,
        n_prefix_rec=None,
        n_prefix_conv=None,
    ):
        super(UHCSGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.n_entity = n_entity
        self.num_dialogues = num_dialogues
        self.num_ml_users = num_ml_users
        self.total_nodes = n_entity + num_dialogues + num_ml_users
        self.num_lightgcn_layers = num_lightgcn_layers

        entity_hidden_size = hidden_size // 2

        # ==========================================
        # KG Encoder (same as MSCRS)
        # ==========================================
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

        # ==========================================
        # UH-CSG: Unified Graph + LightGCN
        # ==========================================
        # Store unified graph edges as buffer (not parameter)
        self.register_buffer("unified_edge_index", unified_edge_index)

        # Precompute self-looped edge index + symmetric normalization coefficients
        # once at init, so each forward pass avoids re-computing add_self_loops/degree.
        # Memory: O(E) instead of materializing O(E×d) message tensors every forward.
        unified_ei_loops, _ = add_self_loops(
            unified_edge_index, num_nodes=self.total_nodes
        )
        row_l, col_l = unified_ei_loops
        deg = degree(col_l, self.total_nodes, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        lgcn_norm = deg_inv_sqrt[row_l] * deg_inv_sqrt[col_l]
        self.register_buffer("_lgcn_edge_index", unified_ei_loops)
        self.register_buffer("_lgcn_norm", lgcn_norm)

        # ==========================================
        # Gated Fusion
        # ==========================================
        self.fusion_gate = nn.Linear(entity_hidden_size * 2, 1)

        # ==========================================
        # Dialogue & User node init mappings
        # ==========================================
        # Pre-compute: for each dialogue node, which entity IDs to average
        # Store as padded tensor + mask for efficient batch computation
        max_items_per_dialogue = max(
            (len(items) for items in dialogue_item_map.values()), default=1
        )
        max_items_per_dialogue = min(max(max_items_per_dialogue, 1), 30)

        dialogue_items_padded = torch.zeros(
            num_dialogues, max_items_per_dialogue, dtype=torch.long
        )
        dialogue_items_mask = torch.zeros(
            num_dialogues, max_items_per_dialogue, dtype=torch.bool
        )

        for j, items in dialogue_item_map.items():
            valid_items = [eid for eid in items if 0 <= eid < n_entity]
            n = len(valid_items)
            if n > 0:
                if n > max_items_per_dialogue:
                    valid_items = valid_items[:max_items_per_dialogue]
                    n = max_items_per_dialogue
                dialogue_items_padded[j, :n] = torch.tensor(
                    valid_items, dtype=torch.long
                )
                dialogue_items_mask[j, :n] = True

        self.register_buffer("dialogue_items_padded", dialogue_items_padded)
        self.register_buffer("dialogue_items_mask", dialogue_items_mask)

        # Pre-compute: for each MovieLens user, which entity IDs to average
        if movielens_edges is not None and movielens_edges.shape[1] > 0:
            user_items = {}
            for i in range(movielens_edges.shape[1]):
                uid = movielens_edges[0, i].item()
                eid = movielens_edges[1, i].item()
                if 0 <= eid < n_entity:
                    if uid not in user_items:
                        user_items[uid] = []
                    user_items[uid].append(eid)

            max_items_per_user = max((len(v) for v in user_items.values()), default=1)
            max_items_per_user = min(max_items_per_user, 5)  # Cap for GPU memory: [K,200,d]=38GB vs [K,5,d]=0.95GB

            ml_items_padded = torch.zeros(
                num_ml_users, max_items_per_user, dtype=torch.long
            )
            ml_items_mask = torch.zeros(
                num_ml_users, max_items_per_user, dtype=torch.bool
            )

            for uid, items in user_items.items():
                if uid < num_ml_users:
                    n = min(len(items), max_items_per_user)
                    ml_items_padded[uid, :n] = torch.tensor(items[:n], dtype=torch.long)
                    ml_items_mask[uid, :n] = True
        else:
            ml_items_padded = torch.zeros(num_ml_users, 1, dtype=torch.long)
            ml_items_mask = torch.zeros(num_ml_users, 1, dtype=torch.bool)

        self.register_buffer("ml_items_padded", ml_items_padded)
        self.register_buffer("ml_items_mask", ml_items_mask)

        # ==========================================
        # Projection layers (same as MSCRS)
        # ==========================================
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

    def _init_dialogue_nodes(self, content_embeds):
        """
        Initialize dialogue node embeddings via mean pooling of their items' content embeddings.
        content_embeds: [N, d] entity embeddings from R-GCN
        Returns: [M, d]
        """
        if self.num_dialogues == 0:
            return torch.zeros(0, content_embeds.size(1), device=content_embeds.device)

        # Gather item embeddings for all dialogues: [M, max_items, d]
        gathered = content_embeds[self.dialogue_items_padded]  # [M, max_items, d]
        # Mask out padding
        mask = self.dialogue_items_mask.unsqueeze(-1).float()  # [M, max_items, 1]
        # Mean pooling (avoid div by 0)
        sum_embeds = (gathered * mask).sum(dim=1)  # [M, d]
        counts = mask.sum(dim=1).clamp(min=1.0)  # [M, 1]
        return sum_embeds / counts

    def _init_ml_user_nodes(self, content_embeds):
        """
        Initialize MovieLens user node embeddings via mean pooling.
        content_embeds: [N, d]
        Returns: [K, d]
        """
        if self.num_ml_users == 0:
            return torch.zeros(0, content_embeds.size(1), device=content_embeds.device)

        gathered = content_embeds[self.ml_items_padded]  # [K, max_items, d]
        mask = self.ml_items_mask.unsqueeze(-1).float()  # [K, max_items, 1]
        sum_embeds = (gathered * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return sum_embeds / counts

    def get_entity_embeds(self):
        """
        UH-CSG entity embedding pipeline:
          1. R-GCN on KG -> content_embeds [N, d]
          2. Init dialogue/user nodes (detached — no grad needed for auxiliary nodes)
          3. LightGCN propagation via sparse matmul (memory: O(E) not O(E×d))
          4. Incremental layer aggregation (no torch.stack → lower peak memory)
          5. Extract item node embeddings [N, d]
          6. Gated fusion: content vs collaborative
          7. Project to hidden_size [N, hidden_size]
        """
        # --- Step 1: R-GCN on KG ---
        node_embeds = self.node_embeds
        content_embeds = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        # content_embeds: [N, entity_hidden_size]

        # --- Step 2: Init auxiliary node embeddings (detached from grad tape) ---
        # Dialogue/user nodes are used only to carry CF signal via graph propagation;
        # there is no loss term that differentiates through them directly.
        content_for_init = content_embeds.detach()
        dialogue_embeds = self._init_dialogue_nodes(content_for_init)  # [M, d]
        ml_user_embeds = self._init_ml_user_nodes(content_for_init)  # [K, d]

        # Concatenate: [N + M + K, d]
        all_embeds = torch.cat([content_embeds, dialogue_embeds, ml_user_embeds], dim=0)

        # --- Step 3 + 4: LightGCN via sparse matmul + incremental mean aggregation ---
        # torch.sparse.mm memory complexity is O(E + N×d) vs O(E×d) for propagate().
        # With ~11M edges and d=256, propagate() would materialise ~11 GB per layer;
        # sparse mm stays below 1 GB regardless of edge count.
        n = self.total_nodes
        A_hat = torch.sparse_coo_tensor(
            self._lgcn_edge_index,
            self._lgcn_norm,
            (n, n),
            device=all_embeds.device,
            dtype=all_embeds.dtype,
        ).coalesce()

        scale = 1.0 / (self.num_lightgcn_layers + 1)
        aggregated = all_embeds * scale  # layer-0 contribution
        current = all_embeds
        for _ in range(self.num_lightgcn_layers):
            current = torch.sparse.mm(A_hat, current)
            aggregated = aggregated + current * scale

        # --- Step 5: Extract item node embeddings only ---
        cf_embeds = aggregated[: self.n_entity]  # [N, d]

        # --- Step 6: Gated fusion ---
        gate_input = torch.cat([content_embeds, cf_embeds], dim=-1)  # [N, 2d]
        gamma = torch.sigmoid(self.fusion_gate(gate_input))  # [N, 1]
        entity_embeds = gamma * cf_embeds + (1 - gamma) * content_embeds  # [N, d]

        # --- Step 7: Project ---
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)  # [N, hidden_size]

        return entity_embeds

    def forward(
        self,
        entity_ids=None,
        token_embeds=None,
        output_entity=False,
        use_rec_prefix=False,
        use_conv_prefix=False,
    ):
        """
        Forward pass — identical interface to MMPrompt.
        Returns: (prompt_embeds, loss_cl)
           or    (prompt_embeds, loss_cl, loss_lb, entity_embeds_all) for deepseek variant
        """
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds_all = entity_embeds  # Save full for rec_logits
            entity_embeds = entity_embeds[
                entity_ids
            ]  # [batch, entity_len, hidden_size]

        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(
                0, 2, 1
            )
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
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

        # No contrastive loss in UH-CSG (placeholder = 0)
        loss_cl = torch.tensor(0.0, device=prompt_embeds.device)
        loss_lb = torch.tensor(0.0, device=prompt_embeds.device)

        # Return format compatible with both original MSCRS and deepseek variants
        # if entity_ids is not None:
        #     return prompt_embeds, loss_cl, loss_lb, entity_embeds_all
        return prompt_embeds, loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        _skip = {"edge", "dialogue_items", "ml_items", "_lgcn_edge_index", "_lgcn_norm"}
        state_dict = {
            k: v
            for k, v in self.state_dict().items()
            if not any(s in k for s in _skip)
        }
        save_path = os.path.join(save_dir, "model.pt")
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, "model.pt")
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device("cpu")), strict=False
        )
        print(
            f"Loaded UHCSGPrompt: missing={missing_keys}, unexpected={unexpected_keys}"
        )
