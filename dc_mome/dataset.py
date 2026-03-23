from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .structures import DCMoMEBatch


def _role_annotate_turns(turns: list[str]) -> list[str]:
    return [("User: " if idx % 2 == 0 else "System: ") + turn for idx, turn in enumerate(turns) if turn]


class _MultimodalFeatureMixin:
    def _load_feature_store(self, path: Path) -> dict[int, np.ndarray]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(key): np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    def _attach_multimodal_features(self, example: dict) -> dict:
        entity_ids = example["entity_ids"]
        example["text_features"] = (
            np.stack(
                [self.text_feature_store.get(entity_id, np.zeros(768, dtype=np.float32)) for entity_id in entity_ids],
                axis=0,
            )
            if entity_ids
            else np.zeros((0, 768), dtype=np.float32)
        )
        example["visual_features"] = (
            np.stack(
                [self.visual_feature_store.get(entity_id, np.zeros(768, dtype=np.float32)) for entity_id in entity_ids],
                axis=0,
            )
            if entity_ids
            else np.zeros((0, 768), dtype=np.float32)
        )
        return example


class _BaseDCMoMEDataset(Dataset, _MultimodalFeatureMixin):
    def __init__(self, config: DataConfig, split: str, phase: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.phase = phase
        self.dataset_dir = config.resolve_phase_data_dir(phase)
        self.multimodal_dir = config.resolve_multimodal_root()
        self.text_feature_store = self._load_feature_store(self.multimodal_dir / "id_embeddings_text.json")
        self.visual_feature_store = self._load_feature_store(self.multimodal_dir / "id_embeddings_image.json")
        self.examples = self._load_examples()

    def _load_examples(self) -> list[dict]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self._attach_multimodal_features(dict(self.examples[index]))


class DCMoMEPretrainDataset(_BaseDCMoMEDataset):
    def __init__(self, config: DataConfig, split: str) -> None:
        super().__init__(config, split, phase="pretrain")

    def _load_examples(self) -> list[dict]:
        data_file = self.dataset_dir / f"{self.split}_data_pretrain.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Missing pretrain file: {data_file}")
        examples: list[dict] = []
        with data_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.config.debug and idx >= 256:
                    break
                dialog = json.loads(line)
                if not dialog.get("rec"):
                    continue
                turns = [utt for utt in dialog.get("context", []) if utt]
                if not turns and not dialog.get("resp"):
                    continue
                role_annotated = _role_annotate_turns(turns)
                resp_role = "System: " if len(turns) % 2 == 0 else "User: "
                response = dialog.get("resp", "")
                if response:
                    role_annotated.append(resp_role + response)
                context_text = " ".join(role_annotated)
                prompt_text = " <sep> ".join(role_annotated)
                base_example = {
                    "turns": role_annotated,
                    "context_text": context_text,
                    "prompt_text": prompt_text,
                    "response_text": response,
                    "entity_ids": dialog.get("entity", [])[-self.config.entity_max_length :],
                }
                for rec_item in dialog.get("rec", []):
                    examples.append(base_example | {"rec_item": rec_item})
        return examples


class DCMoMEAlignmentDataset(_BaseDCMoMEDataset):
    def __init__(self, config: DataConfig, split: str) -> None:
        super().__init__(config, split, phase="alignment")

    def _load_examples(self) -> list[dict]:
        item_ids_path = self.dataset_dir / "item_ids.json"
        if not item_ids_path.exists():
            raise FileNotFoundError(f"Missing item ids file: {item_ids_path}")
        with item_ids_path.open("r", encoding="utf-8") as f:
            item_ids = json.load(f)

        examples: list[dict] = []
        for idx, item_id in enumerate(item_ids):
            if self.config.debug and idx >= 256:
                break
            # Alignment is item-centric, so one item per example is sufficient.
            examples.append(
                {
                    "turns": ["System: item profile"],
                    "context_text": "System: item profile",
                    "prompt_text": "System: item profile",
                    "response_text": "",
                    "entity_ids": [item_id],
                    "rec_item": item_id,
                }
            )
        return examples


class DCMoMERecDataset(_BaseDCMoMEDataset):
    def __init__(self, config: DataConfig, split: str) -> None:
        super().__init__(config, split, phase="recommendation")

    def _load_examples(self) -> list[dict]:
        candidates = [
            self.dataset_dir / f"{self.split}_data_train.jsonl",
            self.dataset_dir / f"{self.split}_data.jsonl",
        ]
        data_file = next((path for path in candidates if path.exists()), None)
        if data_file is None:
            raise FileNotFoundError(f"Missing recommendation file for split={self.split} under {self.dataset_dir}")
        examples: list[dict] = []
        with data_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.config.debug and idx >= 256:
                    break
                dialog = json.loads(line)
                if not dialog.get("rec"):
                    continue
                turns = [utt for utt in dialog.get("context", []) if utt]
                if not turns:
                    continue
                role_annotated = _role_annotate_turns(turns)
                context_text = " ".join(role_annotated)
                prompt_text = " <sep> ".join(role_annotated)
                base_example = {
                    "turns": role_annotated,
                    "context_text": context_text,
                    "prompt_text": prompt_text,
                    "response_text": dialog.get("resp", ""),
                    "entity_ids": dialog.get("entity", [])[-self.config.entity_max_length :],
                }
                for rec_item in dialog.get("rec", []):
                    examples.append(base_example | {"rec_item": rec_item})
        return examples


class DCMoMEConvDataset(_BaseDCMoMEDataset):
    def __init__(self, config: DataConfig, split: str) -> None:
        super().__init__(config, split, phase="conversation")

    def _load_examples(self) -> list[dict]:
        candidates = [
            self.dataset_dir / f"{self.split}_data_processed.jsonl",
            self.dataset_dir / f"{self.split}_data_process.jsonl",
            self.dataset_dir / f"{self.split}_data_conv.jsonl",
            self.dataset_dir / f"{self.split}_data.jsonl",
        ]
        data_file = next((path for path in candidates if path.exists()), None)
        if data_file is None:
            raise FileNotFoundError(
                f"Missing conversation file for split={self.split} under {self.dataset_dir}. "
                f"Expected one of: {', '.join(path.name for path in candidates)}"
            )
        examples: list[dict] = []
        with data_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.config.debug and idx >= 256:
                    break
                dialog = json.loads(line)
                turns = [utt for utt in dialog.get("context", []) if utt]
                response = dialog.get("resp", "")
                if not turns or not response:
                    continue
                role_annotated = _role_annotate_turns(turns)
                retrieved_examples = self._build_retrieved_examples(dialog)
                examples.append(
                    {
                        "turns": role_annotated,
                        "context_text": " ".join(role_annotated),
                        "prompt_text": " <sep> ".join(role_annotated),
                        "response_text": response,
                        "entity_ids": dialog.get("entity", [])[-self.config.entity_max_length :],
                        "rec_item": dialog.get("rec", [None])[0] if dialog.get("rec") else -100,
                        "retrieved_prompt_texts": retrieved_examples,
                    }
                )
        return examples

    def _build_retrieved_examples(self, dialog: dict) -> list[str]:
        prompt_token = "<mask>"
        sep_token = " </s> "
        examples: list[str] = []
        mm_contexts = dialog.get("mm_contexts") or []
        mm_resps = dialog.get("mm_resps") or []
        for context, resp in list(zip(mm_contexts, mm_resps))[: self.config.n_examples]:
            if not context or not resp or resp == "nan":
                continue
            demo = f"{' '.join([prompt_token] * self.config.prompt_max_length)}{sep_token}{context.strip()}{sep_token}System: {str(resp).strip()}"
            examples.append(demo)
        while len(examples) < self.config.n_examples:
            examples.append(" ".join([prompt_token] * (self.config.prompt_max_length + 1)))
        return examples[: self.config.n_examples]


def _pad_entity_payload(examples: list[dict], pad_entity_id: int) -> tuple[list[list[int]], list[list[int]], np.ndarray, np.ndarray]:
    max_entity_len = max((len(example["entity_ids"]) for example in examples), default=0)
    max_entity_len = max(max_entity_len, 1)
    entity_ids = []
    entity_mask = []
    text_features = []
    visual_features = []
    for example in examples:
        padded_ids = example["entity_ids"] + [pad_entity_id] * (max_entity_len - len(example["entity_ids"]))
        mask = [1] * len(example["entity_ids"]) + [0] * (max_entity_len - len(example["entity_ids"]))
        entity_ids.append(padded_ids)
        entity_mask.append(mask)
        text_feature = np.zeros((max_entity_len, 768), dtype=np.float32)
        visual_feature = np.zeros((max_entity_len, 768), dtype=np.float32)
        if len(example["entity_ids"]) > 0:
            text_feature[: len(example["entity_ids"])] = example["text_features"]
            visual_feature[: len(example["entity_ids"])] = example["visual_features"]
        text_features.append(text_feature)
        visual_features.append(visual_feature)
    return entity_ids, entity_mask, np.stack(text_features), np.stack(visual_features)


class DCMoMERecDataCollator:
    def __init__(
        self,
        tokenizer,
        prompt_tokenizer,
        turn_tokenizer,
        pad_entity_id: int,
        device: torch.device,
        context_max_length: int,
        prompt_max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.turn_tokenizer = turn_tokenizer
        self.pad_entity_id = pad_entity_id
        self.device = device
        self.context_max_length = context_max_length
        self.prompt_max_length = prompt_max_length

    def __call__(self, examples: list[dict]) -> DCMoMEBatch:
        contexts = [example["context_text"] for example in examples]
        prompts = [example["prompt_text"] for example in examples]
        current_turns = [example["turns"][-1] for example in examples]
        previous_turns = [" ".join(example["turns"][:-1]) if len(example["turns"]) > 1 else "" for example in examples]
        rec_labels = [example["rec_item"] for example in examples]
        entity_ids, entity_mask, text_features, visual_features = _pad_entity_payload(examples, self.pad_entity_id)

        context_batch = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)
        prompt_batch = self.prompt_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
            return_tensors="pt",
        ).to(self.device)
        current_turn_batch = self.turn_tokenizer(
            current_turns,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)
        previous_turn_batch = self.turn_tokenizer(
            previous_turns,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)

        return DCMoMEBatch(
            context_input_ids=context_batch.input_ids,
            context_attention_mask=context_batch.attention_mask,
            prompt_input_ids=prompt_batch.input_ids,
            prompt_attention_mask=prompt_batch.attention_mask,
            entity_ids=torch.as_tensor(entity_ids, device=self.device),
            entity_mask=torch.as_tensor(entity_mask, device=self.device),
            current_turn_input_ids=current_turn_batch.input_ids,
            current_turn_attention_mask=current_turn_batch.attention_mask,
            previous_turn_input_ids=previous_turn_batch.input_ids,
            previous_turn_attention_mask=previous_turn_batch.attention_mask,
            text_features=torch.as_tensor(text_features, device=self.device),
            visual_features=torch.as_tensor(visual_features, device=self.device),
            rec_labels=torch.as_tensor(rec_labels, device=self.device),
            labels=None,
        )


class DCMoMEAlignmentDataCollator(DCMoMERecDataCollator):
    pass


class DCMoMEConvDataCollator:
    def __init__(
        self,
        tokenizer,
        prompt_tokenizer,
        turn_tokenizer,
        pad_entity_id: int,
        device: torch.device,
        context_max_length: int,
        prompt_max_length: int,
        response_max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.turn_tokenizer = turn_tokenizer
        self.pad_entity_id = pad_entity_id
        self.device = device
        self.context_max_length = context_max_length
        self.prompt_max_length = prompt_max_length
        self.response_max_length = response_max_length

    def __call__(self, examples: list[dict]) -> DCMoMEBatch:
        contexts = [example["context_text"] for example in examples]
        prompts = [example["prompt_text"] for example in examples]
        current_turns = [example["turns"][-1] for example in examples]
        previous_turns = [" ".join(example["turns"][:-1]) if len(example["turns"]) > 1 else "" for example in examples]
        responses = [f"System: {example['response_text']}".strip() for example in examples]
        retrieved_prompts = [prompt for example in examples for prompt in example.get("retrieved_prompt_texts", [])]
        entity_ids, entity_mask, text_features, visual_features = _pad_entity_payload(examples, self.pad_entity_id)

        context_batch = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)
        prompt_batch = self.prompt_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
            return_tensors="pt",
        ).to(self.device)
        current_turn_batch = self.turn_tokenizer(
            current_turns,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)
        previous_turn_batch = self.turn_tokenizer(
            previous_turns,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)

        teacher_inputs = self.tokenizer(
            [f"{context} {response}".strip() for context, response in zip(contexts, responses)],
            padding=True,
            truncation=True,
            max_length=self.context_max_length + self.response_max_length,
            return_tensors="pt",
        ).to(self.device)
        generation_inputs = self.tokenizer(
            [f"{context} System:".strip() for context in contexts],
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)
        response_batch = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.response_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        retrieved_prompt_batch = self.prompt_tokenizer(
            retrieved_prompts,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        ).to(self.device)

        conversation_labels = teacher_inputs.input_ids.clone()
        prompt_lengths = generation_inputs.attention_mask.sum(dim=1)
        for idx, prompt_len in enumerate(prompt_lengths.tolist()):
            conversation_labels[idx, :prompt_len] = -100
        conversation_labels = conversation_labels.masked_fill(teacher_inputs.attention_mask == 0, -100)

        return DCMoMEBatch(
            context_input_ids=context_batch.input_ids,
            context_attention_mask=context_batch.attention_mask,
            prompt_input_ids=prompt_batch.input_ids,
            prompt_attention_mask=prompt_batch.attention_mask,
            entity_ids=torch.as_tensor(entity_ids, device=self.device),
            entity_mask=torch.as_tensor(entity_mask, device=self.device),
            current_turn_input_ids=current_turn_batch.input_ids,
            current_turn_attention_mask=current_turn_batch.attention_mask,
            previous_turn_input_ids=previous_turn_batch.input_ids,
            previous_turn_attention_mask=previous_turn_batch.attention_mask,
            text_features=torch.as_tensor(text_features, device=self.device),
            visual_features=torch.as_tensor(visual_features, device=self.device),
            conversation_input_ids=teacher_inputs.input_ids,
            conversation_attention_mask=teacher_inputs.attention_mask,
            conversation_labels=conversation_labels,
            retrieved_prompt_input_ids=retrieved_prompt_batch.input_ids,
            retrieved_prompt_attention_mask=retrieved_prompt_batch.attention_mask,
            generation_input_ids=generation_inputs.input_ids,
            generation_attention_mask=generation_inputs.attention_mask,
            response_input_ids=response_batch,
            rec_labels=None,
            labels=response_batch,
        )


def build_phase_dataset(config: DataConfig, split: str, phase: str) -> Dataset:
    if phase == "alignment":
        return DCMoMEAlignmentDataset(config, split)
    if phase == "pretrain":
        return DCMoMEPretrainDataset(config, split)
    if phase == "recommendation":
        return DCMoMERecDataset(config, split)
    if phase == "conversation":
        return DCMoMEConvDataset(config, split)
    raise ValueError(f"Unsupported phase for dataset build: {phase}")


def build_phase_collator(
    phase: str,
    tokenizer,
    prompt_tokenizer,
    turn_tokenizer,
    pad_entity_id: int,
    device: torch.device,
    config: DataConfig,
):
    if phase == "alignment":
        return DCMoMEAlignmentDataCollator(
            tokenizer, prompt_tokenizer, turn_tokenizer, pad_entity_id, device,
            config.context_max_length, config.prompt_max_length,
        )
    if phase in {"pretrain", "recommendation"}:
        return DCMoMERecDataCollator(
            tokenizer, prompt_tokenizer, turn_tokenizer, pad_entity_id, device,
            config.context_max_length, config.prompt_max_length,
        )
    if phase == "conversation":
        return DCMoMEConvDataCollator(
            tokenizer, prompt_tokenizer, turn_tokenizer, pad_entity_id, device,
            config.context_max_length, config.prompt_max_length, config.response_max_length,
        )
    raise ValueError(f"Unsupported phase for collator build: {phase}")
