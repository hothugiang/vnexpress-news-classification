import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class KGPrompt(nn.Module):
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
        n_prefix_rec=None,
        n_prefix_conv=None,
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

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
        node_embeds = self.node_embeds
        entity_embeds = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(
        self,
        entity_ids=None,
        token_embeds=None,
        output_entity=False,
        use_rec_prefix=False,
        use_conv_prefix=False,
    ):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[
                entity_ids
            ]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = (
                self.token_proj1(token_embeds) + token_embeds
            )  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(
                0, 2, 1
            )  # (batch_size, token_len, entity_len)
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
        ).permute(
            2, 3, 0, 4, 1, 5
        )  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

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


class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr="add")  # "Add" aggregation.

    def forward(self, x, edge_index):
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算节点度数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 消息传递（邻居信息聚合）
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 直接返回聚合后的输出
        return aggr_out


class MMPrompt_inspired(nn.Module):
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
        edge_index_t_s,
        edge_index_i_s,
        idx_to_id,
        n_prefix_rec=None,
        n_prefix_conv=None,
    ):
        super(MMPrompt_inspired, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor(
            [self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long
        )
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor(
            [self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long
        )
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(
            entity_hidden_size,
            entity_hidden_size,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c, requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s, requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s, requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
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
        node_embeds = self.node_embeds
        entity_embeds = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features, self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1, self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2, self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 + movie_embeds_ts2) / 2

        movie_embeds_is1 = self.conv_is1(node_features, self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1, self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2, self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1 + movie_embeds_is2) / 2
        movie_embeds_mean_t = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        entity_embeds_c1 = self.conv_c1(entity_embeds, self.edge_index_c)
        entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)

        entity_embeds = (
            entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3 + entity_embeds
        ) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[: len(movie_embeds_mean_t)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(
        self,
        entity_ids=None,
        token_embeds=None,
        output_entity=False,
        use_rec_prefix=False,
        use_conv_prefix=False,
    ):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]
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


class MMPrompt(nn.Module):
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
        edge_index_t_s,
        edge_index_i_s,
        idx_to_id,
        n_prefix_rec=None,
        n_prefix_conv=None,
    ):
        super(MMPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor(
            [self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long
        )
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor(
            [self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long
        )

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(
            entity_hidden_size,
            entity_hidden_size,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c, requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s, requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s, requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
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
        node_embeds = self.node_embeds
        entity_embeds = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features, self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1, self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2, self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 + movie_embeds_ts2) / 2

        movie_embeds_is1 = self.conv_is1(node_features, self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1, self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2, self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1 + movie_embeds_is2) / 2
        movie_embeds_mean_t = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        entity_embeds_c1 = self.conv_c1(node_embeds, self.edge_index_c)
        entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)

        entity_embeds = (
            entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3 + entity_embeds
        ) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[: len(movie_embeds_mean_t)]
        # entity_embeds.index_add_(0, indices, movie_embeds_mean_t)

        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(
        self,
        entity_ids=None,
        token_embeds=None,
        output_entity=False,
        use_rec_prefix=False,
        use_conv_prefix=False,
    ):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]
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

    def get_kg_embeds(self, entity_ids):
        """entity_indices: [batch_size, entity_len] (tensor)"""
        node_embeds = self.node_embeds
        e_kg = self.kg_encoder(node_embeds, self.edge_index, self.edge_type)
        e_kg = e_kg + node_embeds
        e_kg = self.entity_proj1(e_kg) + e_kg  # giữ nguyên projection cũ
        # e_kg = self.entity_proj2(e_kg)  # đưa về hidden_size
        # kg_emb shape: [n_entity, hidden_size]
        return e_kg[entity_ids]  # [batch_size, entity_len, hidden_size]

    def get_text_embeds(self, entity_ids):
        real_ids = self.id_to_idx_tensor[entity_ids]
        text_embeds = self.text_embeddings[real_ids]
        return text_embeds

    def get_visual_embeds(self, entity_ids):
        real_ids = self.id_to_idx_tensor[entity_ids]
        image_embeds = self.image_embeddings[real_ids]
        return image_embeds

    def get_entity_embeds(self) -> torch.Tensor:
        """
        Trả về bảng embedding [n_entity, hidden_size] dùng làm item scoring matrix
        cho PromptGPT2forCRS: rec_logits = hidden_state @ entity_embeds.T

        - Items (id_to_idx_tensor >= 0): fuse KG + text + visual (uniform routing 1/3 mỗi)
        - Non-items (id_to_idx_tensor == -1): chỉ dùng KG embedding
        """
        node_embeds = self.node_embeds
        e_kg_all = (
            self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        )
        e_kg_all = (
            self.entity_proj1(e_kg_all) + e_kg_all
        )  # [n_entity, entity_hidden_size]
        h_kg_all = self.kg_proj(e_kg_all)  # [n_entity, hidden_size]

        entity_embeds_out = h_kg_all.clone()

        item_mask = self.id_to_idx_tensor >= 0  # [n_entity] bool
        item_kg_ids = torch.where(item_mask)[0]  # KG indices của items
        item_emb_rows = self.id_to_idx_tensor[item_kg_ids]  # embedding row indices

        if item_kg_ids.numel() > 0:
            h_txt = self.txt_proj(
                self.text_embeddings[item_emb_rows]
            )  # [n_items, hidden]
            h_vis = self.vis_proj(
                self.image_embeddings[item_emb_rows]
            )  # [n_items, hidden]
            h_kg = h_kg_all[item_kg_ids]  # [n_items, hidden]
            # Uniform routing (1/3) vì không có dialogue context
            fused = (h_kg + h_txt + h_vis) / 3.0
            entity_embeds_out[item_kg_ids] = fused

        return entity_embeds_out  # [n_entity, hidden_size]

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
        if state is None:
            state = RoutingState(batch_size, entity_len)

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            e_kg = self.get_kg_embeds(entity_ids=entity_ids)
            e_txt = self.get_text_embeds(entity_ids=entity_ids)
            e_vis = self.get_visual_embeds(entity_ids=entity_ids)

        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds
            token_embeds = self.token_proj2(token_embeds)
            d_t = token_embeds.mean(dim=1)  # (batch, hidden)
        else:
            d_t = torch.zeros(
                batch_size, self.hidden_size, device=next(self.parameters()).device
            )
        routing_emb = dialogue_emb if dialogue_emb is not None else d_t  # [B, hidden]

        h_kg = self.kg_proj(e_kg)
        h_txt = self.txt_proj(e_txt)
        h_vis = self.vis_proj(e_vis)

        o_kg = self.expert_kg(h_kg)  # expert outputs
        o_txt = self.expert_txt(h_txt)
        o_vis = self.expert_vis(h_vis)
        experts = torch.stack([o_kg, o_txt, o_vis], dim=2)

        g_hat, momentum, g = self.compute_routing(
            h_kg, h_txt, h_vis, routing_emb, state
        )
        entity_embeds = (g_hat.unsqueeze(-1) * experts).sum(dim=2)  # [B, L, hidden]

        # ===== Load Balancing Loss (báo cáo §6, Eq. 22) =====
        # L_lb = N_experts × Σ_k f_k × P_k  (Shazeer et al. 2017 / Switch Transformer)
        # Dùng soft routing probabilities g (trước drift) để giữ differentiability
        # FIX 11: tính loss_lb và lưu để trả về
        num_experts = 3
        P_k = g.mean(dim=[0, 1])  # [3] — avg routing prob cho mỗi expert
        f_k = (g == g.max(dim=-1, keepdim=True).values).float().mean(dim=[0, 1])  # [3]
        loss_lb = num_experts * (f_k * P_k).sum()

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

        return prompt_embeds, loss_cl, loss_lb

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
