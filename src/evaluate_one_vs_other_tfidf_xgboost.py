import argparse
import csv
import json
import sys
from pathlib import Path

from train_one_vs_other_tfidf_xgboost import build_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_dependencies():
    """Import joblib/sklearn metrics và báo lỗi rõ nếu môi trường chưa cài đủ."""
    try:
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ModuleNotFoundError as exc:
        missing_package = exc.name
        raise SystemExit(
            f"Thiếu thư viện '{missing_package}'. "
            "Cài bằng: pip install scikit-learn joblib xgboost"
        ) from exc

    return joblib, accuracy_score, precision_score, recall_score, f1_score


def resolve_model_dir(model_dir: Path) -> Path:
    """Ưu tiên thư mục người dùng truyền vào, fallback theo cấu trúc project hiện tại."""
    if model_dir.exists():
        return model_dir

    fallback_dir = PROJECT_ROOT / "models" / "one_vs_other_tfidf_xgboost"
    if model_dir == PROJECT_ROOT / "data" / "models" and fallback_dir.exists():
        print(
            f"Không thấy {model_dir}; dùng fallback {fallback_dir}.",
            file=sys.stderr,
        )
        return fallback_dir

    raise FileNotFoundError(f"Không tìm thấy thư mục model: {model_dir}")


def load_rows_from_csv(csv_file: Path, required_columns):
    """Đọc một file CSV test và kiểm tra các cột bắt buộc."""
    rows = []

    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy test file: {csv_file}")

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing_columns = set(required_columns) - set(fieldnames)
        if missing_columns:
            raise ValueError(
                f"File {csv_file} thiếu cột: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Test set {csv_file} không có dòng dữ liệu nào")

    return rows


def load_category_test_files(test_dir: Path):
    """Đọc các file test one-vs-other đã tách theo category trong data/test."""
    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    csv_files = sorted(path for path in test_dir.glob("*.csv") if path.name != "summary.json")
    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV test nào trong {test_dir}")

    rows_by_stem = {}
    for csv_file in csv_files:
        rows_by_stem[csv_file.stem] = load_rows_from_csv(
            csv_file=csv_file,
            required_columns=["binary_label", "target_category"],
        )

    return rows_by_stem


def load_category_map(model_dir: Path, test_dir: Path | None = None):
    """Map tên file/model stem sang tên category tiếng Việt."""
    category_map = {}

    if test_dir is not None:
        test_summary_path = test_dir / "summary.json"
        if test_summary_path.exists():
            with test_summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            for item in summary.get("files", []):
                file_name = item.get("file")
                category = item.get("target_category") or item.get("category")
                if file_name and category:
                    category_map[Path(file_name).stem] = category

    model_summary_path = model_dir / "summary.json"
    if model_summary_path.exists():
        with model_summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        for item in summary.get("models", []):
            category = item.get("target_category")
            model_path = item.get("model")
            csv_file = item.get("file")

            if category and model_path:
                category_map[Path(model_path).stem] = category
            if category and csv_file:
                category_map[Path(csv_file).stem] = category

    return category_map


def evaluate_model(model_path, target_category, texts, y_true, target_test_file):
    """Tính metric nhị phân cho một classifier target_category vs other."""
    (
        joblib,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    ) = import_dependencies()

    model = joblib.load(model_path)
    y_pred = [int(prediction) for prediction in model.predict(texts)]

    positive_support = sum(y_true)
    negative_support = len(y_true) - positive_support
    predicted_positive = sum(y_pred)
    predicted_negative = len(y_pred) - predicted_positive

    return {
        "model_file": model_path.name,
        "target_category": target_category,
        "target_test_file": target_test_file,
        "test_total": len(y_true),
        "positive_support": positive_support,
        "negative_support": negative_support,
        "predicted_positive": predicted_positive,
        "predicted_negative": predicted_negative,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def write_csv(output_path: Path, rows):
    """Ghi danh sách metric ra CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_file",
        "target_category",
        "target_test_file",
        "test_total",
        "positive_support",
        "negative_support",
        "predicted_positive",
        "predicted_negative",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_binary_test_set(rows_by_stem, target_stem):
    """Lấy đúng file one-vs-other test tương ứng với model."""
    if target_stem not in rows_by_stem:
        available_files = ", ".join(f"{stem}.csv" for stem in sorted(rows_by_stem))
        raise ValueError(
            f"Không tìm thấy test file cho target '{target_stem}.csv'. "
            f"Các file hiện có: {available_files}"
        )

    return rows_by_stem[target_stem]


def evaluate_all_models(model_dir, test_dir, test_file, output_dir, label_column):
    """Đánh giá toàn bộ model one-vs-other trên test set theo từng category."""
    model_dir = resolve_model_dir(model_dir)
    model_paths = sorted(model_dir.glob("*.joblib"))

    if not model_paths:
        raise ValueError(f"Không tìm thấy file .joblib nào trong {model_dir}")

    rows_by_stem = None
    common_test_rows = None
    if test_file is not None:
        common_test_rows = load_rows_from_csv(
            csv_file=test_file,
            required_columns=[label_column],
        )
        category_map = load_category_map(model_dir)
    else:
        rows_by_stem = load_category_test_files(test_dir)
        category_map = load_category_map(model_dir, test_dir)

    metrics = []
    for model_path in model_paths:
        target_category = category_map.get(model_path.stem)
        if not target_category:
            raise ValueError(
                f"Không xác định được target category cho model {model_path.name}. "
                "Kiểm tra summary.json trong thư mục model."
            )

        print(f"Evaluating {model_path.name}: {target_category} vs Other ...")
        if common_test_rows is not None:
            test_rows = common_test_rows
            y_true = [
                1 if (row.get(label_column) or "").strip() == target_category else 0
                for row in test_rows
            ]
            target_test_file = str(test_file)
        else:
            test_rows = get_binary_test_set(
                rows_by_stem=rows_by_stem,
                target_stem=model_path.stem,
            )
            y_true = [int(row["binary_label"]) for row in test_rows]
            target_test_file = str(test_dir / f"{model_path.stem}.csv")

        texts = [build_text(row) for row in test_rows]
        metrics.append(
            evaluate_model(
                model_path=model_path,
                target_category=target_category,
                texts=texts,
                y_true=y_true,
                target_test_file=target_test_file,
            )
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
                "test_dir": str(test_dir) if test_file is None else None,
                "test_file": str(test_file) if test_file is not None else None,
                "label_column": label_column,
                "model_count": len(model_paths),
                "model_type": "TF-IDF + XGBoost",
                "test_strategy": (
                    "per-category one-vs-other files: each model is evaluated "
                    "on its matching test CSV using binary_label"
                    if test_file is None
                    else "single common test file"
                ),
                "outputs": {
                    "csv": str(csv_path),
                    "json": str(json_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return metrics


def parse_args():
    """Đọc tham số dòng lệnh cho script đánh giá 14 binary classifiers."""
    parser = argparse.ArgumentParser(
        description="Evaluate one-vs-other TF-IDF + XGBoost models."
    )
    parser.add_argument(
        "--model-dir",
        default=PROJECT_ROOT / "models" / "one_vs_other_tfidf_xgboost",
        type=Path,
        help="Folder chứa 14 model .joblib.",
    )
    parser.add_argument(
        "--test-dir",
        default=PROJECT_ROOT / "data" / "test",
        type=Path,
        help="Folder chứa test set đã tách theo category.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        type=Path,
        help="Tùy chọn: dùng một file test chung thay vì data/test.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "data" / "reports_tfidf_xgboost",
        type=Path,
        help="Folder lưu kết quả metric.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Tên cột nhãn category trong test set.",
    )
    return parser.parse_args()


def main():
    """Hàm chính khi chạy trực tiếp file này từ dòng lệnh."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    metrics = evaluate_all_models(
        model_dir=args.model_dir,
        test_dir=args.test_dir,
        test_file=args.test_file,
        output_dir=args.output_dir,
        label_column=args.label_column,
    )

    print(f"Done. Evaluated {len(metrics)} models.")
    print(f"Reports saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
