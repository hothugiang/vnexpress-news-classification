"""
dataset_rec_conv.py  (multi-label version)
===========================================
Conversation-aware dataset cho recommendation training.

Thiết kế cốt lõi:
  - Mỗi sample = 1 full conversation (list[turn_dict])
  - 1 turn_pos = 1 turn_dict DUY NHẤT, dù có nhiều rec items
    → "rec" là List[int], KHÔNG expand thành nhiều samples
  - RoutingState cập nhật đúng 1 lần per time-step thực
  - Loss = mean CE over tất cả rec items của turn
  - Evaluation: any-hit — đúng nếu BẤT KỲ item nào trong top-K
"""

import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from utils import padded_tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CRSRecDataset(Dataset):
    """
    Mỗi sample = 1 conversation = list[turn_dict] đã sắp xếp theo turn_pos.

    turn_dict = {
        "context" : List[int]   — tokenized context ids (truncated)
        "prompt"  : List[int]   — tokenized prompt ids  (truncated + CLS prepended)
        "entity"  : List[int]   — entity ids trong context tại turn này
        "rec"     : List[int]   — item ids cần recommend (có thể rỗng)
        "turn_pos": int         — vị trí turn trong hội thoại (dùng để sort)
    }

    Chỉ giữ các conversation có ÍT NHẤT 1 turn với rec label.
    Tất cả turn (cả turn không có rec) đều được giữ lại để
    RoutingState có thể cập nhật momentum liên tục.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset: str,
        split: str,
        tokenizer,
        debug: bool = False,
        context_max_length: int = None,
        entity_max_length: int = None,
        prompt_tokenizer=None,
        prompt_max_length: int = None,
        use_resp=False,
    ):
        super().__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp

        self.context_max_length = context_max_length or tokenizer.model_max_length
        self.entity_max_length = entity_max_length or tokenizer.model_max_length
        # -1 để chừa chỗ cho [CLS] token sẽ được prepend
        self.prompt_max_length = (
            prompt_max_length or prompt_tokenizer.model_max_length
        ) - 1

        data_file = os.path.join(dataset_dir, dataset, f"{split}_data_train.jsonl")
        self.conversations: list[list[dict]] = []
        self._prepare_data(data_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_dialog(self, dialog: dict) -> dict | None:
        """
        Chuyển 1 dialog dict (1 dòng JSONL) → turn dict.
        Trả None nếu context rỗng (bỏ qua).
        """
        context = ""
        prompt_context = ""
        for i, utt in enumerate(dialog["context"]):
            if utt == "":
                continue
            prefix = "User: " if i % 2 == 0 else "System: "
            context += prefix + utt + self.tokenizer.eos_token
            prompt_context += prefix + utt + self.prompt_tokenizer.sep_token

        if not context:
            return None

        ctx_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )[-self.context_max_length :]

        pmt_ids = self.prompt_tokenizer.convert_tokens_to_ids(
            self.prompt_tokenizer.tokenize(prompt_context)
        )[-self.prompt_max_length :]
        pmt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

        return {
            "context": ctx_ids,
            "prompt": pmt_ids,
            "entity": dialog["entity"][-self.entity_max_length :],
            "rec": dialog["rec"],  # list[int], có thể []
            "turn_pos": len(dialog["context"]),  # vị trí để sort
        }

    def _prepare_data(self, data_file: str):
        """
        Đọc toàn bộ file JSONL, nhóm theo conv_id, sort và lọc.
        """
        conv_buffer: dict[int, list[dict]] = defaultdict(list)

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            for line in tqdm(lines, desc="Loading conversations"):
                dialog = json.loads(line)

                # Bỏ qua turn đầu tiên hoàn toàn rỗng (chỉ có [""])
                if len(dialog["context"]) == 1 and dialog["context"][0] == "":
                    continue

                turn = self._tokenize_dialog(dialog)
                if turn is None:
                    continue

                conv_buffer[dialog["conv_id"]].append(turn)

        # Sort từng conversation theo thứ tự thời gian
        kept = 0
        for conv_id, turns in conv_buffer.items():
            turns.sort(key=lambda x: x["turn_pos"])
            # Chỉ giữ conversation có ít nhất 1 turn với rec label
            if any(len(t["rec"]) > 0 for t in turns):
                self.conversations.append(turns)
                kept += 1

        print(
            f"[CRSRecConvDataset] {len(conv_buffer)} total convs → "
            f"{kept} kept (with rec), "
            f"{sum(len(c) for c in self.conversations)} total turns"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> list[dict]:
        return self.conversations[idx]

    def __len__(self) -> int:
        return len(self.conversations)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class CRSRecDataCollator:
    """
    Nhận batch = list of conversations, mỗi conversation = list[turn_dict].

    Trả ra: List[turn_batch_dict] có độ dài = max_turns trong batch.

    Mỗi turn_batch_dict = {
        "context"    : dict   — tokenizer-padded, shape [B, seq_len]
        "prompt"     : dict   — tokenizer-padded, shape [B, prompt_len]
        "entity"     : Tensor — padded, shape     [B, entity_len]
        "rec_labels" : List[List[int]] — rec items per sample ([] nếu không có)
        "valid_mask" : BoolTensor [B] — True nếu conversation còn active ở turn t
        "has_rec"    : BoolTensor [B] — True nếu sample có rec label tại turn t
    }

    rec_labels[i] là List[int] → training loop tính mean CE,
    evaluation loop dùng any-hit.
    """

    def __init__(
        self,
        tokenizer,
        device,
        pad_entity_id: int,
        use_amp: bool = False,
        debug: bool = False,
        context_max_length: int = None,
        entity_max_length: int = None,
        prompt_tokenizer=None,
        prompt_max_length: int = None,
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.pad_entity_id = pad_entity_id
        self.padding = "max_length" if debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length or tokenizer.model_max_length
        self.prompt_max_length = prompt_max_length or prompt_tokenizer.model_max_length
        self.entity_max_length = entity_max_length or tokenizer.model_max_length

    # ------------------------------------------------------------------
    # Internal: pad một list turn_dict thành batched tensors
    # ------------------------------------------------------------------

    def _collate_turns(self, turn_list: list[dict]) -> tuple:
        """
        Nhận list[turn_dict] (mỗi dict là 1 sample tại turn_pos t),
        pad và stack thành tensors.
        """
        ctx_batch = defaultdict(list)
        pmt_batch = defaultdict(list)
        ent_batch = []

        # rec_labels_full : List[List[int]] — dùng cho multilabel loss trong train
        # rec_labels_flat : List[int]       — dùng cho eval/metric, nhất quán baseline
        #                                     (-1 nếu turn không có rec)
        rec_labels_full = [t["rec"] for t in turn_list]
        rec_labels_flat = [t["rec"][0] if t["rec"] else -1 for t in turn_list]

        for t in turn_list:
            ctx_batch["input_ids"].append(t["context"])
            pmt_batch["input_ids"].append(t["prompt"])
            ent_batch.append(t["entity"])

        ctx_out = self.tokenizer.pad(
            ctx_batch,
            padding=self.padding,
            max_length=self.context_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        ctx_out["rec_labels"] = rec_labels_flat
        for k, v in ctx_out.items():
            if not isinstance(v, torch.Tensor):
                ctx_out[k] = torch.as_tensor(v, device=self.device)

        pmt_out = self.prompt_tokenizer.pad(
            pmt_batch,
            padding=self.padding,
            max_length=self.prompt_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        for k, v in pmt_out.items():
            if not isinstance(v, torch.Tensor):
                pmt_out[k] = torch.as_tensor(v, device=self.device)

        ent_tensor = padded_tensor(
            ent_batch,
            pad_idx=self.pad_entity_id,
            pad_tail=True,
            device=self.device,
        )
        return ctx_out, pmt_out, ent_tensor, rec_labels_full

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, conv_batch: list[list[dict]]) -> list[dict]:
        """
        conv_batch[i] = conversation i = list[turn_dict]

        Trả ra: List[turn_batch_dict], độ dài = max_turns
        """
        B = len(conv_batch)
        max_turns = max(len(conv) for conv in conv_batch)

        turn_batches: list[dict] = []

        for t in range(max_turns):
            valid_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
            turns_at_t: list[dict] = []

            for i, conv in enumerate(conv_batch):
                if t < len(conv):
                    turns_at_t.append(conv[t])
                    valid_mask[i] = True
                else:
                    # Lặp lại turn cuối cùng làm padding (dummy).
                    # valid_mask[i] = False → loss bị loại, state không
                    # bị corrupt vì input giống turn trước (routing ổn định).
                    turns_at_t.append(conv[-1])

            ctx, pmt, ent, rec_labels_full = self._collate_turns(turns_at_t)

            # has_rec: conversation còn active VÀ có rec label tại turn t
            has_rec = torch.tensor(
                [
                    valid_mask[i].item() and len(rec_labels_full[i]) > 0
                    for i in range(B)
                ],
                dtype=torch.bool,
                device=self.device,
            )

            turn_batches.append(
                {
                    "context": ctx,
                    "prompt": pmt,
                    "entity": ent,
                    "rec_labels": rec_labels_full,  # List[int], length B
                    "valid_mask": valid_mask,  # [B] — active conversations
                    "has_rec": has_rec,  # [B] — có rec label tại t
                }
            )

        return turn_batches


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import gpt2_special_tokens_dict, prompt_special_tokens_dict

    debug, device = True, torch.device("cuda")
    dataset_dir, dataset = "rec_data", "inspired"

    tokenizer = AutoTokenizer.from_pretrained("models/DialoGPT-small")
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained("models/roberta_base")
    prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    ds = CRSRecDataset(
        dataset_dir=dataset_dir,
        dataset=dataset,
        split="test",
        tokenizer=tokenizer,
        debug=debug,
        prompt_tokenizer=prompt_tokenizer,
    )
    print(f"Conversations: {len(ds)}")
    for i, t in enumerate(ds[0]):
        print(f"  turn {i:2d} | rec={t['rec']} | n_entity={len(t['entity'])}")

    from dataset_dbpedia_inspired import DBpedia

    kg = DBpedia(
        dataset_dir=dataset_dir, dataset=dataset, debug=debug
    ).get_entity_kg_info()
    collator = CRSRecDataCollator(
        tokenizer=tokenizer,
        device=device,
        pad_entity_id=kg["pad_entity_id"],
        prompt_tokenizer=prompt_tokenizer,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)
    for turn_batches in loader:
        for t, tb in enumerate(turn_batches):
            print(
                f"t={t} | ent={tb['entity'].shape} | "
                f"has_rec={tb['has_rec'].tolist()} | rec={tb['rec_labels']}"
            )
        break
