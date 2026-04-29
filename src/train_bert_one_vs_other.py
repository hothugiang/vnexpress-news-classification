"""
Fine-tune BERT (PhoBERT / XLM-RoBERTa) cho 14 binary one-vs-other classifiers.

Giữ nguyên flow: mỗi file data/one_vs_other/<category>.csv → 1 model BERT binary.
Thay thế TF-IDF + Logistic Regression bằng BERT fine-tuning.

Usage:
    python src/train_bert_one_vs_other.py \
        --input-dir data/one_vs_other \
        --output-dir models/bert_one_vs_other \
        --model-name vinai/phobert-base-v2 \
        --epochs 3 \
        --batch-size 32 \
        --lr 2e-5 \
        --max-len 128 \
        --seed 42

Để train một category cụ thể:
    --category suc_khoe
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_imports():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Thiếu thư viện '{exc.name}'. Chạy: uv sync"
        ) from exc
    return torch, DataLoader, Dataset, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup


def build_text(row):
    """Ghép tiêu đề và mô tả ngắn thành input text (giống baseline)."""
    title = (row.get("title_clean") or row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    return f"{title} {description}".strip()


def load_binary_dataset(input_file: Path):
    """Đọc một file one-vs-other CSV, trả về texts, labels, target_category."""
    texts, labels = [], []
    target_category = None

    with input_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"binary_label", "target_category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"File {input_file} thiếu cột: {', '.join(sorted(missing))}")
        for row in reader:
            texts.append(build_text(row))
            labels.append(int(row["binary_label"]))
            target_category = row["target_category"]

    if target_category is None:
        raise ValueError(f"File {input_file} không có dòng dữ liệu nào")

    return texts, labels, target_category


class BinaryDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_train_epoch(model, loader, optimizer, scheduler, device, torch_mod):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch_mod.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(loader)


def run_eval(model, loader, device):
    import torch
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def train_one_model(input_file: Path, output_dir: Path, args):
    """Train một BERT binary classifier cho một file one-vs-other."""
    torch, DataLoader, Dataset, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup = load_imports()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels, target_category = load_binary_dataset(input_file)
    positive = sum(labels)
    negative = len(labels) - positive

    # Val split: 10% cuối để monitor, train trên 90% đầu
    split_idx = int(len(texts) * 0.9)
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    val_texts, val_labels = texts[split_idx:], labels[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = BinaryDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = BinaryDataset(val_texts, val_labels, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0.0
    model_save_dir = output_dir / input_file.stem

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, scheduler, device, torch)
        val_metrics = run_eval(model, val_loader, device)

        print(
            f"    epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} f1={val_metrics['f1']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

    # Lưu meta cho evaluate script
    meta = {
        "target_category": target_category,
        "model_name": args.model_name,
        "max_len": args.max_len,
        "best_val_f1": best_f1,
        "positive": positive,
        "negative": negative,
        "total": len(texts),
    }
    with (model_save_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Giải phóng VRAM giữa các model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Saved: {model_save_dir}  (best_val_f1={best_f1:.4f})")
    return {
        "file": input_file.name,
        "model_dir": str(model_save_dir),
        "target_category": target_category,
        "positive": positive,
        "negative": negative,
        "total": len(texts),
        "best_val_f1": best_f1,
    }


def train_all_models(input_dir: Path, output_dir: Path, args):
    """Train 14 binary BERT classifiers từ toàn bộ file CSV trong data/one_vs_other."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {input_dir}")

    if args.category:
        csv_files = [f for f in csv_files if f.stem == args.category]
        if not csv_files:
            raise ValueError(f"Không tìm thấy file cho category: {args.category}")

    summary = []
    for input_file in csv_files:
        print(f"\nTraining {input_file.name} ...")
        item = train_one_model(input_file, output_dir, args)
        summary.append(item)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "model_type": "BERT binary one-vs-other",
                "text_input": "title_clean/title + description",
                "max_len": args.max_len,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "models": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT cho 14 bộ one-vs-other (thay thế TF-IDF + LR)."
    )
    parser.add_argument("--input-dir", default=PROJECT_ROOT / "data" / "one_vs_other", type=Path)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "models" / "bert_one_vs_other", type=Path)
    parser.add_argument(
        "--model-name", default="vinai/phobert-base-v2",
        help="HuggingFace model ID. Gợi ý: vinai/phobert-base-v2 | xlm-roberta-base",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--max-len", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--category", default=None,
        help="Chỉ train một category cụ thể (tên file không có .csv), ví dụ: suc_khoe",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    summary = train_all_models(args.input_dir, args.output_dir, args)
    print(f"\nDone. Trained {len(summary)} models in {args.output_dir}")


if __name__ == "__main__":
    main()
