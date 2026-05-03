import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_common_dependencies():
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Thiếu thư viện '{exc.name}'. Cài bằng: pip install scikit-learn joblib"
        ) from exc

    return (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )


def load_test_rows(test_file: Path, label_column: str):
    rows = []
    if not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy test file: {test_file}")

    with test_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if label_column not in fieldnames:
            raise ValueError(f"File {test_file} thiếu cột '{label_column}'")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Test file {test_file} không có dữ liệu")
    return rows


def build_text(row):
    title = (row.get("title_clean") or row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    return f"{title} {description}".strip()


def load_tfidf_or_xgb_models(model_dir: Path):
    import joblib

    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Không tìm thấy summary.json trong {model_dir}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    models = []
    for item in summary.get("models", []):
        model_path = Path(item["model"])
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        # Một số summary cũ lưu path model không còn khớp với cấu trúc thư mục hiện tại.
        # Fallback về đúng thư mục model-dir đang evaluate, giữ nguyên tên file.
        if not model_path.exists():
            fallback_path = model_dir / Path(item["model"]).name
            if fallback_path.exists():
                model_path = fallback_path

        models.append(
            {
                "stem": model_path.stem,
                "target_category": item["target_category"],
                "model_obj": joblib.load(model_path),
            }
        )

    if not models:
        raise ValueError(f"Không tìm thấy model nào trong {model_dir}")
    return models


def score_tfidf_or_xgb(models, texts):
    score_by_category = {}
    for item in models:
        probabilities = item["model_obj"].predict_proba(texts)
        score_by_category[item["target_category"]] = [float(row[1]) for row in probabilities]
    return score_by_category


def load_bert_dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Thiếu thư viện '{exc.name}'. Cài bằng: pip install torch transformers sentencepiece"
        ) from exc

    return torch, DataLoader, Dataset, AutoTokenizer, AutoModelForSequenceClassification


class BertDataset:
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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


def load_bert_model_dirs(model_dir: Path):
    model_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not model_dirs:
        raise ValueError(f"Không tìm thấy BERT model dirs trong {model_dir}")

    items = []
    for path in model_dirs:
        with (path / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        items.append(
            {
                "path": path,
                "target_category": meta["target_category"],
                "max_len": meta["max_len"],
            }
        )
    return items


def score_bert(model_dir: Path, texts, batch_size: int):
    torch, DataLoader, Dataset, AutoTokenizer, AutoModelForSequenceClassification = load_bert_dependencies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_by_category = {}

    for item in load_bert_model_dirs(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(str(item["path"]))
        model = AutoModelForSequenceClassification.from_pretrained(str(item["path"])).to(device)
        model.eval()

        dataset = BertDataset(texts, tokenizer, item["max_len"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        probabilities = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                probabilities.extend(probs.cpu().tolist())

        score_by_category[item["target_category"]] = [float(p) for p in probabilities]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return score_by_category


def predict_final_labels(score_by_category, categories, threshold: float, other_label: str):
    predictions = []
    num_rows = len(next(iter(score_by_category.values())))
    for row_idx in range(num_rows):
        best_category = None
        best_score = -1.0
        scores = {}
        for category in categories:
            score = score_by_category[category][row_idx]
            scores[category] = score
            if score > best_score:
                best_score = score
                best_category = category
        final_category = best_category if best_score >= threshold else other_label
        predictions.append((final_category, best_score, scores, best_category))
    return predictions


def write_predictions_csv(output_path: Path, rows, categories):
    fieldnames = [
        "true_category",
        "predicted_category",
        "predicted_score",
        "raw_best_category",
        "threshold_applied",
    ] + [f"score_{category}" for category in categories]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_argmax(
    model_type: str,
    model_dir: Path,
    test_file: Path,
    output_dir: Path,
    label_column: str,
    bert_batch_size: int,
    threshold: float,
    other_label: str,
):
    rows = load_test_rows(test_file, label_column)
    texts = [build_text(row) for row in rows]
    y_true = [(row.get(label_column) or "").strip() for row in rows]

    if model_type == "tfidf_lr":
        models = load_tfidf_or_xgb_models(model_dir)
        categories = [item["target_category"] for item in models]
        score_by_category = score_tfidf_or_xgb(models, texts)
    elif model_type == "tfidf_xgboost":
        models = load_tfidf_or_xgb_models(model_dir)
        categories = [item["target_category"] for item in models]
        score_by_category = score_tfidf_or_xgb(models, texts)
    elif model_type == "bert":
        score_by_category = score_bert(model_dir, texts, bert_batch_size)
        categories = sorted(score_by_category)
    else:
        raise ValueError(f"Model type không hợp lệ: {model_type}")

    predictions = predict_final_labels(score_by_category, categories, threshold, other_label)
    y_pred = [item[0] for item in predictions]
    evaluation_labels = list(dict.fromkeys(categories + [other_label] + sorted(set(y_true))))

    (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    ) = import_common_dependencies()

    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for true_category, prediction in zip(y_true, predictions):
        pred_category, pred_score, score_map, raw_best_category = prediction
        row = {
            "true_category": true_category,
            "predicted_category": pred_category,
            "predicted_score": pred_score,
            "raw_best_category": raw_best_category,
            "threshold_applied": int(pred_category != raw_best_category),
        }
        for category in categories:
            row[f"score_{category}"] = score_map[category]
        prediction_rows.append(row)

    predictions_csv = output_dir / "predictions.csv"
    write_predictions_csv(predictions_csv, prediction_rows, categories)

    class_report_dict = classification_report(
        y_true,
        y_pred,
        labels=evaluation_labels,
        output_dict=True,
        zero_division=0,
    )
    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=evaluation_labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=evaluation_labels)
    confusion_csv = output_dir / "confusion_matrix.csv"
    with confusion_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + evaluation_labels)
        for category, row in zip(evaluation_labels, cm.tolist()):
            writer.writerow([category] + row)

    metrics = {
        "model_type": model_type,
        "stage_2_method": "max_voting_threshold",
        "test_file": str(test_file),
        "num_test_rows": len(rows),
        "threshold": threshold,
        "other_label": other_label,
        "threshold_fallback_count": sum(1 for item in predictions if item[0] != item[3]),
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(class_report_dict, f, ensure_ascii=False, indent=2)
    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(class_report_text)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "stage_2_method": "max_voting_threshold",
                "model_dir": str(model_dir),
                "test_file": str(test_file),
                "threshold": threshold,
                "other_label": other_label,
                "outputs": {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "classification_report_json": str(output_dir / "classification_report.json"),
                    "classification_report_txt": str(output_dir / "classification_report.txt"),
                    "confusion_matrix_csv": str(confusion_csv),
                    "predictions_csv": str(predictions_csv),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done. Saved final argmax evaluation to: {output_dir}")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate final multiclass baseline by argmax over 14 binary classifiers."
    )
    parser.add_argument(
        "--model-type",
        choices=["tfidf_lr", "tfidf_xgboost", "bert"],
        required=True,
        help="Họ model của 14 binary classifiers.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Thư mục model. Nếu bỏ trống sẽ dùng default theo model-type.",
    )
    parser.add_argument(
        "--test-file",
        default=PROJECT_ROOT / "data" / "split" / "test.csv",
        type=Path,
        help="Test set gốc đa lớp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục lưu report. Nếu bỏ trống sẽ dùng default theo model-type.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Tên cột nhãn đa lớp trong test set.",
    )
    parser.add_argument(
        "--bert-batch-size",
        default=64,
        type=int,
        help="Batch size cho BERT khi predict final stage 2 argmax.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Nếu score cao nhất < threshold thì gán về nhãn Khác.",
    )
    parser.add_argument(
        "--other-label",
        default="Khác",
        help="Nhãn fallback khi không model nào vượt threshold.",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    default_model_dirs = {
        "tfidf_lr": PROJECT_ROOT / "models" / "one_vs_other_tfidf_lr",
        "tfidf_xgboost": PROJECT_ROOT / "models" / "one_vs_other_tfidf_xgboost",
        "bert": PROJECT_ROOT / "models" / "bert_one_vs_other",
    }
    default_output_dirs = {
        "tfidf_lr": PROJECT_ROOT / "data" / "final_reports" / "tfidf_lr_maxvoting",
        "tfidf_xgboost": PROJECT_ROOT / "data" / "final_reports" / "tfidf_xgboost_maxvoting",
        "bert": PROJECT_ROOT / "data" / "final_reports" / "bert_maxvoting",
    }

    model_dir = args.model_dir or default_model_dirs[args.model_type]
    output_dir = args.output_dir or default_output_dirs[args.model_type]

    evaluate_argmax(
        model_type=args.model_type,
        model_dir=model_dir,
        test_file=args.test_file,
        output_dir=output_dir,
        label_column=args.label_column,
        bert_batch_size=args.bert_batch_size,
        threshold=args.threshold,
        other_label=args.other_label,
    )


if __name__ == "__main__":
    main()
