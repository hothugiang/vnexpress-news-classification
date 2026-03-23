from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    dataset: str = "inspired"
    rec_data_root: Path = Path("rec_data")
    conv_data_root: Path = Path("conv_data")
    context_max_length: int = 200
    response_max_length: int = 128
    prompt_max_length: int = 200
    entity_max_length: int = 32
    debug: bool = False
    generation_max_length: int = 50
    n_examples: int = 3

    def resolve_rec_data_dir(self) -> Path:
        return self.rec_data_root / self.dataset

    def resolve_conv_data_dir(self) -> Path:
        return self.conv_data_root / self.dataset

    def resolve_graph_data_dir(self) -> Path:
        return self.resolve_rec_data_dir()

    def resolve_multimodal_root(self) -> Path:
        return self.resolve_rec_data_dir()

    def resolve_phase_data_dir(self, phase: str) -> Path:
        if phase == "conversation":
            return self.resolve_conv_data_dir()
        return self.resolve_rec_data_dir()


@dataclass(slots=True)
class PromptConfig:
    hidden_size: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_block: int = 2
    n_prefix_rec: int = 10
    n_prefix_conv: int = 20


@dataclass(slots=True)
class EncoderConfig:
    kg_dim: int = 384
    text_dim: int = 768
    visual_dim: int = 768
    dialogue_dim: int = 768
    projector_hidden_dim: int = 768
    expert_hidden_dim: int = 512
    router_key_dim: int = 64
    num_bases: int = 8
    beta: float = 0.5
    temperature: float = 0.07


@dataclass(slots=True)
class TrainingConfig:
    output_dir: Path = Path("output/dc_mome")
    lm_model_name_or_path: str = "models/DialoGPT-small"
    text_model_name_or_path: str = "models/roberta_base"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 5
    num_warmup_steps: int = 0
    balance_loss_weight: float = 0.01
    align_loss_weight: float = 1.0
    freeze_backbone: bool = True
    phase: str = "alignment"


@dataclass(slots=True)
class DCMoMEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
