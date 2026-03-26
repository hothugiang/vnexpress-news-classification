from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .config import DCMoMEConfig
from .dataset import build_phase_collator, build_phase_dataset
from .evaluators import ConvEvaluator, RecEvaluator
from .graphs import load_mscrs_kg
from .losses import (
    generation_loss,
    load_balancing_loss,
    multimodal_alignment_loss,
    recommendation_ce_loss,
)
from .pipeline import DCMoMEModel
from .prompt_gpt2 import PromptGPT2forCRS

GPT2_SPECIAL_TOKENS_DICT = {
    "pad_token": "<pad>",
    "additional_special_tokens": ["<movie>"],
}

PROMPT_SPECIAL_TOKENS_REC = {
    "additional_special_tokens": ["<movie>"],
}

PROMPT_SPECIAL_TOKENS_CONV = {
    "additional_special_tokens": ["<movie>", "<mask>"],
}


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    name: str
    task: str
    use_rec_prefix: bool = False
    use_conv_prefix: bool = False
    requires_pretrained_prompt: bool = False
    prompt_output_entity: bool = True


class TrainLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.log_dir / "train.log"
        self.jsonl_path = self.log_dir / "train.jsonl"

    def log(self, event: str, **payload) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with self.text_path.open("a", encoding="utf-8") as f:
            pretty = " ".join(f"{key}={value}" for key, value in record.items())
            f.write(pretty + "\n")
        print(f"[dc_mome] {event} " + " ".join(f"{k}={v}" for k, v in payload.items()))


PHASE_SPECS: dict[str, PhaseSpec] = {
    "alignment": PhaseSpec(name="alignment", task="alignment"),
    "pretrain": PhaseSpec(name="pretrain", task="pretrain"),
    "recommendation": PhaseSpec(
        name="recommendation",
        task="recommendation",
        use_rec_prefix=True,
        requires_pretrained_prompt=True,
    ),
    "conversation": PhaseSpec(
        name="conversation",
        task="conversation",
        use_conv_prefix=True,
        requires_pretrained_prompt=True,
        prompt_output_entity=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="inspired")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--rec-data-root", type=str, default="rec_data")
    parser.add_argument("--conv-data-root", type=str, default="conv_data")
    parser.add_argument("--phase", type=str, default="alignment")
    parser.add_argument("--phases", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="output/dc_mome")
    parser.add_argument(
        "--lm-model-name-or-path", type=str, default="models/DialoGPT-small"
    )
    parser.add_argument(
        "--text-model-name-or-path", type=str, default="models/roberta_base"
    )
    parser.add_argument("--save-every-phase", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DCMoMEConfig:
    config = DCMoMEConfig()
    config.data.dataset = args.dataset
    rec_root = args.rec_data_root if args.dataset_root is None else args.dataset_root
    config.data.rec_data_root = config.data.rec_data_root.__class__(rec_root)
    config.data.conv_data_root = config.data.conv_data_root.__class__(
        args.conv_data_root
    )
    config.training.phase = args.phase
    config.training.batch_size = args.batch_size
    config.training.eval_batch_size = args.eval_batch_size
    config.training.num_epochs = args.num_epochs
    config.training.output_dir = Path(args.output_dir)
    config.training.lm_model_name_or_path = args.lm_model_name_or_path
    config.training.text_model_name_or_path = args.text_model_name_or_path
    return config


def resolve_phase_order(args: argparse.Namespace) -> list[PhaseSpec]:
    phase_names = args.phases.split() if args.phases else [args.phase]
    resolved = []
    for phase_name in phase_names:
        if phase_name not in PHASE_SPECS:
            raise ValueError(f"Unsupported phase: {phase_name}")
        resolved.append(PHASE_SPECS[phase_name])
    return resolved


def build_dataloaders(
    config: DCMoMEConfig,
    device: torch.device,
    pad_entity_id: int,
) -> dict[str, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(config.training.lm_model_name_or_path)
    tokenizer.add_special_tokens(GPT2_SPECIAL_TOKENS_DICT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = GPT2_SPECIAL_TOKENS_DICT["pad_token"]
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        config.training.text_model_name_or_path
    )
    prompt_special_tokens = (
        PROMPT_SPECIAL_TOKENS_CONV
        if config.training.phase == "conversation"
        else PROMPT_SPECIAL_TOKENS_REC
    )
    prompt_tokenizer.add_special_tokens(prompt_special_tokens)
    turn_tokenizer = prompt_tokenizer
    collator = build_phase_collator(
        config.training.phase,
        tokenizer,
        prompt_tokenizer,
        turn_tokenizer,
        pad_entity_id,
        device,
        config.data,
    )

    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "valid", "test"):
        dataset = build_phase_dataset(
            config.data, split=split, phase=config.training.phase
        )
        batch_size = (
            config.training.batch_size
            if split == "train"
            else config.training.eval_batch_size
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
        )
    return dataloaders


def recommendation_prompt_learning(
    prompt_model: PromptGPT2forCRS | None,
    dc_mome_model: DCMoMEModel,
    outputs,
    batch,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if prompt_model is None:
        return None, None
    entity_embeds, candidate_ids = dc_mome_model.get_all_entity_embeds()
    if candidate_ids.numel() == 0:
        return None, None
    rec_outputs = prompt_model(
        input_ids=batch.context_input_ids,
        attention_mask=batch.context_attention_mask,
        prompt_embeds=outputs.prompt_embeds,
        rec=True,
        entity_embeds=entity_embeds,
        rec_labels=batch.rec_labels,
        return_dict=True,
    )
    return rec_outputs.rec_logits, candidate_ids


def compute_phase_loss(
    outputs,
    batch,
    config: DCMoMEConfig,
    phase_spec: PhaseSpec,
    prompt_model: PromptGPT2forCRS | None = None,
    dc_mome_model: DCMoMEModel | None = None,
) -> torch.Tensor:
    kg_mask, text_mask, visual_mask = modality_presence_masks(batch, dc_mome_model)
    align = multimodal_alignment_loss(
        outputs.h_kg,
        outputs.h_t,
        outputs.h_v,
        kg_mask=kg_mask,
        text_mask=text_mask,
        visual_mask=visual_mask,
        temperature=config.encoder.temperature,
    )
    balance = config.training.balance_loss_weight * load_balancing_loss(
        outputs.routing_weights
    )
    if phase_spec.task == "alignment":
        return config.training.align_loss_weight * align
    if phase_spec.task == "pretrain":
        rec_logits, candidate_ids = recommendation_prompt_learning(
            prompt_model, dc_mome_model, outputs, batch
        )
        rec_loss = recommendation_ce_loss(rec_logits, batch.rec_labels)
        return balance + rec_loss + config.training.align_loss_weight * align
    if phase_spec.task == "recommendation":
        rec_logits, candidate_ids = recommendation_prompt_learning(
            prompt_model, dc_mome_model, outputs, batch
        )
        rec_loss = recommendation_ce_loss(rec_logits, batch.rec_labels)
        return balance + rec_loss + config.training.align_loss_weight * align
    if phase_spec.task == "conversation":
        lm_loss = conversation_lm_loss(
            prompt_model,
            outputs.prompt_token_embeds,
            batch,
            dc_mome_model=dc_mome_model,
        )
        return (
            balance
            + generation_loss(lm_loss)
            + config.training.align_loss_weight * align
        )
    raise ValueError(f"Unsupported phase task: {phase_spec.task}")


def modality_presence_masks(
    batch,
    dc_mome_model: DCMoMEModel | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kg_mask = batch.entity_mask.float()
    if dc_mome_model is None:
        text_mask = torch.zeros_like(kg_mask)
        visual_mask = torch.zeros_like(kg_mask)
        return kg_mask, text_mask, visual_mask
    text_presence, visual_presence = dc_mome_model.lookup_modality_presence(batch.entity_ids)
    text_mask = kg_mask * text_presence.float()
    visual_mask = kg_mask * visual_presence.float()
    return kg_mask, text_mask, visual_mask


def conversation_lm_loss(
    prompt_model: PromptGPT2forCRS | None,
    prompt_tokens: torch.Tensor,
    batch,
    dc_mome_model: DCMoMEModel | None = None,
) -> torch.Tensor | None:
    if (
        prompt_model is None
        or dc_mome_model is None
        or batch.conversation_input_ids is None
        or batch.conversation_labels is None
    ):
        return None
    inputs_embeds, attention_mask, _ = build_mapped_conversation_inputs(
        prompt_model,
        dc_mome_model,
        prompt_tokens,
        batch.conversation_input_ids,
        batch.conversation_attention_mask,
        batch.entity_ids,
    )
    labels = pad_conversation_labels(batch.conversation_labels, prompt_tokens)
    return prompt_model(
        inputs_embeds=inputs_embeds,
        input_ids=None,
        attention_mask=attention_mask,
        conv=True,
        conv_labels=labels,
        return_dict=True,
    ).conv_loss


def generate_responses(
    prompt_model: PromptGPT2forCRS,
    prompt_tokens: torch.Tensor,
    batch,
    max_new_tokens: int,
    dc_mome_model: DCMoMEModel | None = None,
) -> torch.Tensor:
    if dc_mome_model is None:
        raise ValueError("dc_mome_model is required for conversation generation")
    inputs_embeds, attention_mask, _ = build_mapped_conversation_inputs(
        prompt_model,
        dc_mome_model,
        prompt_tokens,
        batch.generation_input_ids,
        batch.generation_attention_mask,
        batch.entity_ids,
    )
    past_key_values = None
    next_input_ids = None
    generated_tokens = []
    for _ in range(max_new_tokens):
        outputs = prompt_model(
            input_ids=next_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            conv=True,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        if (next_token == prompt_model.config.eos_token_id).all():
            break
        past_key_values = outputs.past_key_values
        next_input_ids = next_token
        inputs_embeds = None
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
    if not generated_tokens:
        return torch.empty(
            (attention_mask.size(0), 0), dtype=torch.long, device=attention_mask.device
        )
    return torch.cat(generated_tokens, dim=1)


def build_mapped_conversation_inputs(
    prompt_model: PromptGPT2forCRS,
    dc_mome_model: DCMoMEModel,
    prompt_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    entity_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_input_embeddings = prompt_model.get_input_embeddings()(input_ids)
    return dc_mome_model.prompt_formatter.augment_context_for_conversation(
        prompt_tokens=prompt_tokens,
        entity_embeds=dc_mome_model.forward_entity_summary(entity_ids),
        word_embeddings=prompt_model.get_input_embeddings().weight,
        context_input_embeddings=context_input_embeddings,
        attention_mask=attention_mask,
        mapping=True,
    )


def pad_conversation_labels(
    labels: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> torch.Tensor:
    prefix = torch.full(
        (labels.size(0), prompt_tokens.size(1)),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([prefix, labels], dim=1)


def evaluate_phase_outputs(
    outputs,
    batch,
    phase_spec: PhaseSpec,
    prompt_model: PromptGPT2forCRS | None,
    dc_mome_model: DCMoMEModel,
    tokenizer,
    metrics: dict,
    config: DCMoMEConfig,
) -> None:
    if phase_spec.task in {"pretrain", "recommendation"}:
        rec_logits, candidate_ids = recommendation_prompt_learning(
            prompt_model, dc_mome_model, outputs, batch
        )
        if rec_logits is None or candidate_ids.numel() == 0:
            return
        if phase_spec.task == "recommendation":
            item_candidate_ids = dc_mome_model.item_ids
            rec_logits = rec_logits[:, item_candidate_ids]
            candidate_ids = item_candidate_ids
        topk = min(50, candidate_ids.numel())
        ranked_idx = rec_logits.topk(topk, dim=-1).indices
        ranked_item_ids = candidate_ids[ranked_idx]
        metrics["rec"].update(ranked_item_ids, batch.rec_labels)
    elif phase_spec.task == "conversation" and prompt_model is not None:
        preds = generate_responses(
            prompt_model,
            outputs.prompt_token_embeds,
            batch,
            config.data.generation_max_length,
            dc_mome_model=dc_mome_model,
        )
        metrics["conv"].update(preds, batch.response_input_ids)


def run_epoch(
    model: DCMoMEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: DCMoMEConfig,
    phase_spec: PhaseSpec,
    prompt_model: PromptGPT2forCRS | None,
    tokenizer,
    logger: TrainLogger | None = None,
    epoch_idx: int | None = None,
    split: str = "train",
    log_interval: int = 50,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    if prompt_model is not None:
        prompt_model.train(is_train and not config.training.freeze_backbone)
    total_loss = 0.0
    total_steps = 0
    metrics = {
        "rec": RecEvaluator(),
        "conv": ConvEvaluator(tokenizer),
    }

    for step_idx, batch in enumerate(dataloader):
        with torch.set_grad_enabled(is_train):
            outputs = model(
                batch,
                previous_momentum=None,
                use_rec_prefix=phase_spec.use_rec_prefix,
                use_conv_prefix=phase_spec.use_conv_prefix,
                prompt_output_entity=phase_spec.prompt_output_entity,
            )
            loss = compute_phase_loss(
                outputs,
                batch,
                config,
                phase_spec,
                prompt_model=prompt_model,
                dc_mome_model=model,
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        evaluate_phase_outputs(
            outputs, batch, phase_spec, prompt_model, model, tokenizer, metrics, config
        )

        total_loss += float(loss.detach())
        total_steps += 1
        if logger is not None and (step_idx == 0 or (step_idx + 1) % log_interval == 0):
            logger.log(
                "step",
                phase=phase_spec.name,
                split=split,
                epoch=epoch_idx,
                step=step_idx,
                loss=round(float(loss.detach()), 6),
                batch_size=int(batch.entity_ids.size(0)),
                h_kg_norm=round(float(outputs.h_kg.norm(dim=-1).mean().detach()), 6),
                h_t_norm=round(float(outputs.h_t.norm(dim=-1).mean().detach()), 6),
                h_v_norm=round(float(outputs.h_v.norm(dim=-1).mean().detach()), 6),
            )

    if total_steps == 0:
        return 0.0, {}
    phase_metrics = {}
    if phase_spec.task in {"pretrain", "recommendation"}:
        phase_metrics = metrics["rec"].report()
    elif phase_spec.task == "conversation":
        phase_metrics = metrics["conv"].report()
    return total_loss / total_steps, phase_metrics


def save_metrics(output_dir: Path, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_phase_checkpoint(
    output_dir: Path, dc_mome_model: DCMoMEModel, prompt_model: PromptGPT2forCRS
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dc_mome_model.save(output_dir / "dc_mome")
    torch.save(prompt_model.state_dict(), output_dir / "prompt_lm.pt")


def load_phase_checkpoint(
    output_dir: Path, dc_mome_model: DCMoMEModel, prompt_model: PromptGPT2forCRS
) -> None:
    dc_mome_model.load(output_dir / "dc_mome")
    prompt_path = output_dir / "prompt_lm.pt"
    if prompt_path.exists():
        prompt_model.load_state_dict(
            torch.load(prompt_path, map_location="cpu"), strict=False
        )


def maybe_load_pretrained_prompt(model: DCMoMEModel, phase_output_dir: Path) -> None:
    pretrain_best_dir = phase_output_dir / "pretrain" / "best"
    if pretrain_best_dir.exists():
        model.load(pretrain_best_dir / "dc_mome")


def run_training(args: argparse.Namespace) -> None:
    config = build_config(args)
    device = torch.device(args.device)
    phase_order = resolve_phase_order(args)
    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(config.training.output_dir)
    logger.log(
        "run_start",
        dataset=config.data.dataset,
        rec_data_root=str(config.data.rec_data_root),
        conv_data_root=str(config.data.conv_data_root),
        phases=[phase.name for phase in phase_order],
        batch_size=config.training.batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_epochs=config.training.num_epochs,
        device=str(device),
    )

    graph_bundle = load_mscrs_kg(config.data.resolve_graph_data_dir())
    prompt_has_conversation = any(phase.name == "conversation" for phase in phase_order)
    text_tokenizer = AutoTokenizer.from_pretrained(config.training.text_model_name_or_path)
    text_tokenizer.add_special_tokens(
        PROMPT_SPECIAL_TOKENS_CONV if prompt_has_conversation else PROMPT_SPECIAL_TOKENS_REC
    )
    dialogue_backbone = AutoModel.from_pretrained(
        config.training.text_model_name_or_path
    )
    dialogue_backbone.resize_token_embeddings(len(text_tokenizer))
    model = DCMoMEModel(config, graph_bundle, dialogue_backbone).to(device)
    lm_tokenizer = AutoTokenizer.from_pretrained(config.training.lm_model_name_or_path)
    lm_tokenizer.add_special_tokens(GPT2_SPECIAL_TOKENS_DICT)
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = GPT2_SPECIAL_TOKENS_DICT["pad_token"]
    prompt_model = PromptGPT2forCRS.from_pretrained(
        config.training.lm_model_name_or_path
    ).to(device)
    prompt_model.resize_token_embeddings(len(lm_tokenizer))
    prompt_model.config.pad_token_id = lm_tokenizer.pad_token_id
    if config.training.freeze_backbone:
        prompt_model.requires_grad_(False)
        model.prompt_text_backbone.requires_grad_(False)

    for phase_spec in phase_order:
        phase_dir = config.training.output_dir / phase_spec.name
        config.training.phase = phase_spec.name
        dataloaders = build_dataloaders(config, device, graph_bundle.pad_entity_id)
        logger.log(
            "phase_start",
            phase=phase_spec.name,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            num_warmup_steps=config.training.num_warmup_steps,
            batch_size=config.training.batch_size,
            eval_batch_size=config.training.eval_batch_size,
            num_epochs=config.training.num_epochs,
            num_train_examples=len(dataloaders["train"].dataset),
            num_valid_examples=len(dataloaders["valid"].dataset),
            num_test_examples=len(dataloaders["test"].dataset),
        )
        if phase_spec.requires_pretrained_prompt:
            maybe_load_pretrained_prompt(model, config.training.output_dir)
            logger.log(
                "checkpoint_load",
                phase=phase_spec.name,
                source=str(config.training.output_dir / "pretrain" / "best"),
            )

        trainable_parameters = list(model.parameters())
        if not config.training.freeze_backbone:
            trainable_parameters += [
                p for p in prompt_model.parameters() if p.requires_grad
            ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        best_valid = float("inf")
        best_epoch = -1
        best_valid_metrics: dict[str, float] = {}
        history: list[dict] = []

        for epoch_idx in range(config.training.num_epochs):
            train_loss, train_metrics = run_epoch(
                model,
                dataloaders["train"],
                optimizer,
                config,
                phase_spec,
                prompt_model,
                lm_tokenizer,
                logger=logger,
                epoch_idx=epoch_idx,
                split="train",
                log_interval=args.log_interval,
            )
            valid_loss, valid_metrics = run_epoch(
                model,
                dataloaders["valid"],
                None,
                config,
                phase_spec,
                prompt_model,
                lm_tokenizer,
                logger=logger,
                epoch_idx=epoch_idx,
                split="valid",
                log_interval=args.log_interval,
            )
            history.append(
                {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                }
            )

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch_idx
                best_valid_metrics = valid_metrics
                save_phase_checkpoint(phase_dir / "best", model, prompt_model)
                logger.log(
                    "checkpoint_save",
                    phase=phase_spec.name,
                    epoch=epoch_idx,
                    checkpoint="best",
                    best_valid_loss=round(best_valid, 6),
                )

            if args.save_every_phase:
                save_phase_checkpoint(
                    phase_dir / f"epoch_{epoch_idx:02d}", model, prompt_model
                )
                logger.log(
                    "checkpoint_save",
                    phase=phase_spec.name,
                    epoch=epoch_idx,
                    checkpoint=f"epoch_{epoch_idx:02d}",
                )

            logger.log(
                "epoch_end",
                phase=phase_spec.name,
                epoch=epoch_idx,
                train_loss=round(train_loss, 6),
                valid_loss=round(valid_loss, 6),
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )

        save_phase_checkpoint(phase_dir / "final", model, prompt_model)
        logger.log("checkpoint_save", phase=phase_spec.name, checkpoint="final")
        load_phase_checkpoint(phase_dir / "best", model, prompt_model)
        test_loss, test_metrics = run_epoch(
            model,
            dataloaders["test"],
            None,
            config,
            phase_spec,
            prompt_model,
            lm_tokenizer,
            logger=logger,
            epoch_idx=best_epoch,
            split="test",
            log_interval=args.log_interval,
        )
        metrics = {
            "phase": phase_spec.name,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid,
            "test_loss": test_loss,
            "best_valid_metrics": best_valid_metrics,
            "test_metrics": test_metrics,
            "num_train_examples": len(dataloaders["train"].dataset),
            "num_valid_examples": len(dataloaders["valid"].dataset),
            "num_test_examples": len(dataloaders["test"].dataset),
            "history": history,
        }
        save_metrics(phase_dir, metrics)
        logger.log(
            "phase_end",
            phase=phase_spec.name,
            best_epoch=best_epoch,
            best_valid_loss=round(best_valid, 6),
            test_loss=round(test_loss, 6),
            test_metrics=test_metrics,
        )
    logger.log("run_end", output_dir=str(config.training.output_dir))


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
