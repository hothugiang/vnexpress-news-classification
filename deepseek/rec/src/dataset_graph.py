"""
UH-CSG Graph Builder
====================
Builds the Unified Heterogeneous Conversational Signal Graph (CSG).

Graph structure:
  - Item nodes (0 .. N-1): all DBpedia entities
  - Dialogue nodes (N .. N+M-1): each training dialogue session
  - MovieLens user nodes (N+M .. N+M+K-1): external CF users

Edge types (all stored in one unified edge_index):
  - E_TS: text similarity edges between items (from text_sim)
  - E_IS: image similarity edges between items (from image_sim)
  - E_DI: dialogue-item edges (bipartite)
  - E_UI: MovieLens user-item edges (bipartite)
"""

import json
import os
import torch
import numpy as np
from collections import defaultdict
from loguru import logger


class UHCSGGraphBuilder:
    """
    Builds the unified CSG graph from existing MSCRS data sources.
    """

    def __init__(
        self,
        n_entity: int,
        edge_index_t_s: torch.Tensor,  # text similarity edges (local idx space)
        edge_index_i_s: torch.Tensor,  # image similarity edges (local idx space)
        idx_to_id: dict,  # local movie idx -> global entity id
        dialogue_items: list = None,  # list of lists: dialogue_items[j] = [entity_ids in dialogue j]
        movielens_edges: torch.Tensor = None,  # [2, num_ml_edges] in (user_local_idx, entity_id)
        num_ml_users: int = 0,
    ):
        self.n_entity = n_entity
        self.idx_to_id = idx_to_id
        self.num_movies = len(idx_to_id)

        # --- Convert text_sim edges from local idx to global entity IDs ---
        ts_src_local = edge_index_t_s[0].tolist()
        ts_dst_local = edge_index_t_s[1].tolist()
        ts_src_global = [idx_to_id[s] for s in ts_src_local]
        ts_dst_global = [idx_to_id[d] for d in ts_dst_local]
        self.ts_edges_global = torch.tensor(
            [ts_src_global, ts_dst_global], dtype=torch.long
        )

        # --- Convert image_sim edges similarly ---
        is_src_local = edge_index_i_s[0].tolist()
        is_dst_local = edge_index_i_s[1].tolist()
        # image_sim may have its own idx_to_id, but typically same movie set
        is_src_global = [idx_to_id.get(s, s) for s in is_src_local]
        is_dst_global = [idx_to_id.get(d, d) for d in is_dst_local]
        self.is_edges_global = torch.tensor(
            [is_src_global, is_dst_global], dtype=torch.long
        )

        # --- Dialogue-item edges ---
        self.dialogue_items = dialogue_items if dialogue_items is not None else []
        self.num_dialogues = len(self.dialogue_items)

        # --- MovieLens edges ---
        self.movielens_edges = movielens_edges
        self.num_ml_users = num_ml_users

        # Total nodes
        self.total_nodes = n_entity + self.num_dialogues + self.num_ml_users

        logger.info(
            f"UH-CSG Graph: {n_entity} entities + {self.num_dialogues} dialogues "
            f"+ {self.num_ml_users} ML users = {self.total_nodes} total nodes"
        )

    def build_unified_edge_index(self):
        """
        Build the unified edge_index [2, total_edges] in global node space.
        All edges are made bidirectional.
        """
        all_src = []
        all_dst = []

        # 1) Text similarity edges (item <-> item, already in global entity space)
        all_src.append(self.ts_edges_global[0])
        all_dst.append(self.ts_edges_global[1])
        # Reverse
        all_src.append(self.ts_edges_global[1])
        all_dst.append(self.ts_edges_global[0])

        # 2) Image similarity edges (item <-> item)
        all_src.append(self.is_edges_global[0])
        all_dst.append(self.is_edges_global[1])
        # Reverse
        all_src.append(self.is_edges_global[1])
        all_dst.append(self.is_edges_global[0])

        # 3) Dialogue-item edges
        dialogue_offset = self.n_entity
        di_src = []
        di_dst = []
        for j, items in enumerate(self.dialogue_items):
            dialogue_node = dialogue_offset + j
            for item_id in items:
                if 0 <= item_id < self.n_entity:
                    di_src.append(dialogue_node)
                    di_dst.append(item_id)
        if di_src:
            di_src_t = torch.tensor(di_src, dtype=torch.long)
            di_dst_t = torch.tensor(di_dst, dtype=torch.long)
            # Bidirectional
            all_src.append(di_src_t)
            all_dst.append(di_dst_t)
            all_src.append(di_dst_t)
            all_dst.append(di_src_t)

        logger.info(
            f"  Dialogue-Item edges: {len(di_src)} (bidirectional: {len(di_src) * 2})"
        )

        # 4) MovieLens user-item edges
        ml_offset = self.n_entity + self.num_dialogues
        if self.movielens_edges is not None and self.movielens_edges.shape[1] > 0:
            ml_user_local = self.movielens_edges[0]  # local user idx
            ml_item_global = self.movielens_edges[1]  # global entity id
            ml_user_global = ml_user_local + ml_offset
            # Bidirectional
            all_src.append(ml_user_global)
            all_dst.append(ml_item_global)
            all_src.append(ml_item_global)
            all_dst.append(ml_user_global)
            logger.info(
                f"  MovieLens edges: {ml_user_local.shape[0]} (bidirectional: {ml_user_local.shape[0] * 2})"
            )

        # Concatenate
        unified_src = torch.cat(all_src, dim=0)
        unified_dst = torch.cat(all_dst, dim=0)
        unified_edge_index = torch.stack([unified_src, unified_dst], dim=0)

        # Remove duplicates
        edge_set = set()
        unique_src = []
        unique_dst = []
        for i in range(unified_edge_index.shape[1]):
            s, d = unified_edge_index[0, i].item(), unified_edge_index[1, i].item()
            if (s, d) not in edge_set:
                edge_set.add((s, d))
                unique_src.append(s)
                unique_dst.append(d)

        unified_edge_index = torch.tensor([unique_src, unique_dst], dtype=torch.long)
        logger.info(
            f"  Total unified edges (deduplicated): {unified_edge_index.shape[1]}"
        )

        return unified_edge_index

    def build_dialogue_item_map(self):
        """
        Returns a dict mapping dialogue_idx -> list of global entity IDs.
        Used for initializing dialogue node embeddings.
        """
        return {j: items for j, items in enumerate(self.dialogue_items)}

    def get_graph_info(self):
        """Returns all info needed by UHCSGPrompt."""
        unified_edge_index = self.build_unified_edge_index()
        dialogue_item_map = self.build_dialogue_item_map()

        return {
            "unified_edge_index": unified_edge_index,
            "num_dialogues": self.num_dialogues,
            "num_ml_users": self.num_ml_users,
            "total_nodes": self.total_nodes,
            "dialogue_item_map": dialogue_item_map,
            "movielens_edges": self.movielens_edges,
        }


def extract_dialogue_items_from_dataset(dataset_dir, dataset, split="train"):
    """
    Extract entity mentions from each training dialogue.

    Reads the processed data files to get entity IDs per dialogue.
    Returns: list of lists, where each inner list contains entity IDs mentioned in that dialogue.
    """
    data_file = os.path.join(dataset_dir, dataset, f"{split}_data_processed.jsonl")

    if not os.path.exists(data_file):
        # Try alternative filename patterns
        alt_files = [
            os.path.join(dataset_dir, dataset, f"{split}_data.jsonl"),
            os.path.join(dataset_dir, dataset, f"{split}_data_train.jsonl"),
        ]
        for alt in alt_files:
            if os.path.exists(alt):
                data_file = alt
                break
    if not os.path.exists(data_file):
        logger.warning(f"Could not find training data file. Tried: {data_file}")
        return []

    dialogue_items = []
    current_dialogue_entities = set()
    current_dialogue_id = None

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Group by dialogue_id (or conv_id)
            dial_id = data.get("conv_id", data.get("dialog_id", None))

            if dial_id != current_dialogue_id:
                if current_dialogue_entities:
                    dialogue_items.append(list(current_dialogue_entities))
                current_dialogue_entities = set()
                current_dialogue_id = dial_id

            # Collect entity IDs from this turn
            entities = data.get("entity", data.get("entities", []))
            if isinstance(entities, list):
                for eid in entities:
                    if isinstance(eid, int):
                        current_dialogue_entities.add(eid)

        # Don't forget last dialogue
        if current_dialogue_entities:
            dialogue_items.append(list(current_dialogue_entities))

    logger.info(f"Extracted entities from {len(dialogue_items)} dialogues")
    return dialogue_items


def load_movielens_from_pt(pt_file: str):
    """
    Load MovieLens edges từ file .pt đã chuẩn bị sẵn.

    File .pt format:
        {
            'edges': torch.LongTensor shape [2, num_edges],
                     row 0 = user_local_idx  (0 .. num_users-1)
                     row 1 = entity_id       (global DBpedia entity ID)
            'num_users': int
        }

    Returns:
        (edges, num_users)
    """
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"MovieLens .pt file not found: {pt_file}")

    data = torch.load(pt_file, map_location="cpu", weights_only=False)

    edges = data["edges"]  # [2, E]
    num_users = data["num_users"]
    ml_entity = data["ml_to_entity"]
    user_id_to_local = data["user_id_to_local"]

    logger.info(
        f"Loaded MovieLens: {pt_file} → "
        f"{num_users} users, {edges.shape[1]} interactions"
        f" (entities covered: {len(set(edges[1].tolist()))})"
        f" (MovieLens entities: {len(ml_entity)})"
        f" (User ID mapping: {len(user_id_to_local)})"
    )
    return edges, num_users, ml_entity, user_id_to_local


# def create_mock_movielens_edges(
#     item_ids, n_users=100, interactions_per_user=20, seed=42
# ):
#     """
#     Create mock MovieLens user-item edges for testing the pipeline.

#     Args:
#         item_ids: list of valid item entity IDs
#         n_users: number of mock users
#         interactions_per_user: average interactions per user
#         seed: random seed

#     Returns:
#         movielens_edges: [2, num_edges] tensor (user_local_idx, item_entity_id)
#         n_users: number of users
#     """
#     rng = np.random.RandomState(seed)
#     item_ids = list(item_ids)

#     src = []
#     dst = []
#     for u in range(n_users):
#         n_items = rng.randint(5, interactions_per_user * 2)
#         selected_items = rng.choice(
#             item_ids, size=min(n_items, len(item_ids)), replace=False
#         )
#         for item_id in selected_items:
#             src.append(u)
#             dst.append(item_id)

#     edges = torch.tensor([src, dst], dtype=torch.long)
#     logger.info(
#         f"Created mock MovieLens edges: {n_users} users, {edges.shape[1]} interactions"
#     )
#     return edges, n_users


# def create_mock_dialogue_items(
#     item_ids, n_dialogues=500, items_per_dialogue=5, seed=42
# ):
#     """
#     Create mock dialogue-item data for testing the pipeline.

#     Args:
#         item_ids: list of valid item entity IDs
#         n_dialogues: number of mock dialogues
#         items_per_dialogue: average items per dialogue

#     Returns:
#         dialogue_items: list of lists of entity IDs
#     """
#     rng = np.random.RandomState(seed)
#     item_ids = list(item_ids)

#     dialogue_items = []
#     for _ in range(n_dialogues):
#         n_items = rng.randint(2, items_per_dialogue * 2)
#         selected = rng.choice(item_ids, size=min(n_items, len(item_ids)), replace=False)
#         dialogue_items.append(selected.tolist())

#     logger.info(f"Created mock dialogue items: {n_dialogues} dialogues")
#     return dialogue_items
