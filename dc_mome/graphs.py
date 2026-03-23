from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .structures import SemanticGraphBundle


def _load_feature_store(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(key): np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def _build_feature_matrix(
    num_entities: int,
    feature_store: dict[int, np.ndarray],
    feature_dim: int,
) -> torch.Tensor:
    matrix = np.zeros((num_entities, feature_dim), dtype=np.float32)
    for entity_id, feature in feature_store.items():
        if 0 <= entity_id < num_entities:
            matrix[entity_id] = feature
    return torch.as_tensor(matrix, dtype=torch.float32)


def load_mscrs_kg(dataset_dir: Path) -> SemanticGraphBundle:
    with (dataset_dir / "dbpedia_subkg.json").open("r", encoding="utf-8") as f:
        entity_kg = json.load(f)
    with (dataset_dir / "entity2id.json").open("r", encoding="utf-8") as f:
        entity2id = json.load(f)
    with (dataset_dir / "relation2id.json").open("r", encoding="utf-8") as f:
        relation2id = json.load(f)
    with (dataset_dir / "item_ids.json").open("r", encoding="utf-8") as f:
        item_ids = json.load(f)

    edge_triplets: set[tuple[int, int, int]] = set()
    for entity_id in entity2id.values():
        for relation_id, tail_id in entity_kg.get(str(entity_id), []):
            edge_triplets.add((entity_id, tail_id, relation_id))
            edge_triplets.add((tail_id, entity_id, relation_id))

    if edge_triplets:
        edge = torch.as_tensor(sorted(edge_triplets), dtype=torch.long)
        edge_index = edge[:, :2].t().contiguous()
        edge_type = edge[:, 2].contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    pad_entity_id = max(entity2id.values()) + 1
    num_entities = pad_entity_id + 1
    text_store = _load_feature_store(dataset_dir / "id_embeddings_text.json")
    visual_store = _load_feature_store(dataset_dir / "id_embeddings_image.json")
    text_dim = next(iter(text_store.values())).shape[-1] if text_store else 768
    visual_dim = next(iter(visual_store.values())).shape[-1] if visual_store else 768

    return SemanticGraphBundle(
        edge_index=edge_index,
        edge_type=edge_type,
        num_entities=num_entities,
        num_relations=len(relation2id),
        pad_entity_id=pad_entity_id,
        item_ids=item_ids,
        text_feature_matrix=_build_feature_matrix(num_entities, text_store, text_dim),
        visual_feature_matrix=_build_feature_matrix(num_entities, visual_store, visual_dim),
    )
