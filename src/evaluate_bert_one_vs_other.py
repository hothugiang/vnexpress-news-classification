"""
Evaluate 14 BERT binary one-vs-other classifiers trên test set.

Output format giống evaluate_one_vs_other_tfidf_lr.py để dễ so sánh.

Usage:
    python src/evaluate_bert_one_vs_other.py \
        --model-dir models/bert_one_vs_other \
        --test-dir data/test \
        --output-dir data/reports_bert
"""

import csv
import json
import sys
import argparse
from pathlib import Path

from train_bert_one_vs_other import build_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_imports():
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Thiếu thư viện '{exc.name}'. Chạy: uv sync") from exc
    return torch, DataLoader, AutoTokenizer, AutoModelForSequenceClassification, accuracy_score, f1_score, precision_score, recall_score


def load_rows_from_csv(csv_file: Path, required_columns):
    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy test file: {csv_file}")

    rows = []
    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"File {csv_file} thiếu cột: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Test file {csv_file} không có dòng dữ liệu nào")
    return rows


def load_category_test_files(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    csv_files = sorted(p for p in test_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV test nào trong {test_dir}")

    rows_by_stem = {}
    for csv_file in csv_files:
        rows_by_stem[csv_file.stem] = load_rows_from_csv(
            csv_file, required_columns=["binary_label", "target_category"]
        )
    return rows_by_stem


class SimpleDataset:
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
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
        }


def predict_binary(model_dir: Path, texts, batch_size, device, torch_mod, DataLoader, AutoTokenizer, AutoModelForSequenceClassification):
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy meta.json trong {model_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    dataset = SimpleDataset(texts, tokenizer, max_len=meta["max_len"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds = []
    with torch_mod.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())

    del model
    import gc; gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()

    return all_preds, meta["target_category"]


def evaluate_model(model_dir: Path, texts, y_true, target_category, target_test_file, batch_size, device, imports):
    torch_mod, DataLoader, AutoTokenizer, AutoModelForSequenceClassification, accuracy_score, f1_score, precision_score, recall_score = imports

    y_pred, _ = predict_binary(model_dir, texts, batch_size, device, torch_mod, DataLoader, AutoTokenizer, AutoModelForSequenceClassification)

    return {
        "model_dir": model_dir.name,
        "target_category": target_category,
        "target_test_file": target_test_file,
        "test_total": len(y_true),
        "positive_support": sum(y_true),
        "negative_support": len(y_true) - sum(y_true),
        "predicted_positive": sum(y_pred),
        "predicted_negative": len(y_pred) - sum(y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def write_csv(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_dir", "target_category", "target_test_file",
        "test_total", "positive_support", "negative_support",
        "predicted_positive", "predicted_negative",
        "accuracy", "precision", "recall", "f1",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all_models(model_dir: Path, test_dir: Path, output_dir: Path, batch_size: int):
    imports = load_imports()
    torch_mod = imports[0]
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not model_dirs:
        raise ValueError(f"Không tìm thấy model nào trong {model_dir}")

    rows_by_stem = load_category_test_files(test_dir)
    metrics = []

    for mdir in model_dirs:
        with (mdir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        target_category = meta["target_category"]
        stem = mdir.name

        if stem not in rows_by_stem:
            print(f"  Bỏ qua {stem}: không tìm thấy test file tương ứng", file=sys.stderr)
            continue

        test_rows = rows_by_stem[stem]
        texts = [build_text(row) for row in test_rows]
        y_true = [int(row["binary_label"]) for row in test_rows]
        target_test_file = str(test_dir / f"{stem}.csv")

        print(f"Evaluating {stem}: {target_category} vs Other ...")
        result = evaluate_model(
            model_dir=mdir,
            texts=texts,
            y_true=y_true,
            target_category=target_category,
            target_test_file=target_test_file,
            batch_size=batch_size,
            device=device,
            imports=imports,
        )
        metrics.append(result)
        print(
            f"  acc={result['accuracy']:.4f} prec={result['precision']:.4f} "
            f"rec={result['recall']:.4f} f1={result['f1']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "individual_binary_metrics.csv"
    json_path = output_dir / "individual_binary_metrics.json"
    summary_path = output_dir / "summary.json"

    write_csv(csv_path, metrics)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "test_dir": str(test_dir),
                "model_type": "BERT binary one-vs-other",
                "model_count": len(metrics),
                "outputs": {"csv": str(csv_path), "json": str(json_path)},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    avg_f1 = sum(m["f1"] for m in metrics) / len(metrics) if metrics else 0
    print(f"\nDone. Avg F1: {avg_f1:.4f} | Reports saved in: {output_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BERT one-vs-other models.")
    parser.add_argument("--model-dir", default=PROJECT_ROOT / "models" / "bert_one_vs_other", type=Path)
    parser.add_argument("--test-dir", default=PROJECT_ROOT / "data" / "test", type=Path)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "data" / "reports_bert", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    evaluate_all_models(args.model_dir, args.test_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
