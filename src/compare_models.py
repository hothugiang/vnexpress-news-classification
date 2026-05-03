import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1")


def load_metrics(csv_path: Path, model_name: str):
    """Đọc một file metrics CSV và chuẩn hóa về cùng schema."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metrics: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        required = {"target_category", *METRIC_COLUMNS}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"File {csv_path} thiếu cột: {', '.join(sorted(missing))}"
            )

        for row in reader:
            normalized = {
                "model_name": model_name,
                "target_category": row["target_category"],
            }
            for metric in METRIC_COLUMNS:
                normalized[metric] = float(row[metric])
            rows.append(normalized)

    if not rows:
        raise ValueError(f"File {csv_path} không có dòng dữ liệu nào")

    return rows


def build_comparison_rows(model_rows_map):
    """Ghép metric của các model theo từng category."""
    all_categories = sorted(
        {
            row["target_category"]
            for rows in model_rows_map.values()
            for row in rows
        }
    )

    per_model_by_category = {
        model_name: {row["target_category"]: row for row in rows}
        for model_name, rows in model_rows_map.items()
    }

    comparison_rows = []
    for category in all_categories:
        row = {"target_category": category}
        for model_name, rows_by_category in per_model_by_category.items():
            metrics = rows_by_category.get(category)
            if metrics is None:
                for metric in METRIC_COLUMNS:
                    row[f"{model_name}_{metric}"] = ""
            else:
                for metric in METRIC_COLUMNS:
                    row[f"{model_name}_{metric}"] = metrics[metric]
        comparison_rows.append(row)

    return comparison_rows


def compute_macro_summary(model_rows_map):
    """Tính macro-average cho từng model."""
    summary = []
    for model_name, rows in model_rows_map.items():
        item = {
            "model_name": model_name,
            "num_categories": len(rows),
        }
        for metric in METRIC_COLUMNS:
            item[f"macro_{metric}"] = sum(row[metric] for row in rows) / len(rows)
        summary.append(item)

    summary.sort(key=lambda row: row["macro_f1"], reverse=True)
    return summary


def write_csv(output_path: Path, rows, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_models(lr_csv: Path, xgb_csv: Path, bert_csv: Path, output_dir: Path):
    model_rows_map = {
        "tfidf_lr": load_metrics(lr_csv, "tfidf_lr"),
        "tfidf_xgboost": load_metrics(xgb_csv, "tfidf_xgboost"),
        "bert": load_metrics(bert_csv, "bert"),
    }

    comparison_rows = build_comparison_rows(model_rows_map)
    macro_summary = compute_macro_summary(model_rows_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_category_csv = output_dir / "per_category_comparison.csv"
    macro_csv = output_dir / "macro_summary.csv"
    summary_json = output_dir / "summary.json"

    per_category_fieldnames = ["target_category"]
    for model_name in model_rows_map:
        for metric in METRIC_COLUMNS:
            per_category_fieldnames.append(f"{model_name}_{metric}")

    write_csv(per_category_csv, comparison_rows, per_category_fieldnames)
    write_csv(
        macro_csv,
        macro_summary,
        ["model_name", "num_categories", "macro_accuracy", "macro_precision", "macro_recall", "macro_f1"],
    )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "tfidf_lr": str(lr_csv),
                    "tfidf_xgboost": str(xgb_csv),
                    "bert": str(bert_csv),
                },
                "metric_columns": list(METRIC_COLUMNS),
                "macro_summary": macro_summary,
                "best_model_by_macro_f1": macro_summary[0]["model_name"] if macro_summary else None,
                "outputs": {
                    "per_category_csv": str(per_category_csv),
                    "macro_csv": str(macro_csv),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return per_category_csv, macro_csv, summary_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF + LR, TF-IDF + XGBoost, and BERT binary model reports."
    )
    parser.add_argument(
        "--lr-csv",
        default=PROJECT_ROOT / "data" / "reports_tfidf_lr" / "individual_binary_metrics.csv",
        type=Path,
        help="Metrics CSV của TF-IDF + Logistic Regression.",
    )
    parser.add_argument(
        "--xgboost-csv",
        default=PROJECT_ROOT / "data" / "reports_tfidf_xgboost" / "individual_binary_metrics.csv",
        type=Path,
        help="Metrics CSV của TF-IDF + XGBoost.",
    )
    parser.add_argument(
        "--bert-csv",
        default=PROJECT_ROOT / "data" / "reports_bert" / "individual_binary_metrics.csv",
        type=Path,
        help="Metrics CSV của BERT.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "data" / "model_comparison",
        type=Path,
        help="Thư mục lưu bảng so sánh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    per_category_csv, macro_csv, summary_json = compare_models(
        lr_csv=args.lr_csv,
        xgb_csv=args.xgboost_csv,
        bert_csv=args.bert_csv,
        output_dir=args.output_dir,
    )
    print(f"Saved per-category comparison: {per_category_csv}")
    print(f"Saved macro summary: {macro_csv}")
    print(f"Saved summary json: {summary_json}")


if __name__ == "__main__":
    main()
