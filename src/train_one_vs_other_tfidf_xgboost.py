import argparse
import csv
import json
import sys
from pathlib import Path


def import_dependencies():
    """Import sklearn/xgboost/joblib và báo lỗi rõ nếu môi trường chưa cài."""
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        missing_package = exc.name
        raise SystemExit(
            f"Thiếu thư viện '{missing_package}'. "
            "Cài bằng: pip install scikit-learn joblib xgboost"
        ) from exc

    return joblib, TfidfVectorizer, FeatureUnion, Pipeline, XGBClassifier


def build_text(row):
    """Ghép tiêu đề và mô tả ngắn thành input text cho mô hình."""
    title = (row.get("title_clean") or row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    return f"{title} {description}".strip()


def load_binary_dataset(input_file: Path):
    """Đọc một file one-vs-other và trả về text, label, target category."""
    texts = []
    labels = []
    target_category = None

    with input_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"binary_label", "target_category"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"File {input_file} thiếu cột: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            texts.append(build_text(row))
            labels.append(int(row["binary_label"]))
            target_category = row["target_category"]

    if target_category is None:
        raise ValueError(f"File {input_file} không có dòng dữ liệu nào")

    return texts, labels, target_category


def build_model(args):
    """Tạo pipeline TF-IDF + XGBoost cho một bộ binary."""
    (
        _joblib,
        TfidfVectorizer,
        FeatureUnion,
        Pipeline,
        XGBClassifier,
    ) = import_dependencies()

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        tree_method=args.tree_method,
        verbosity=1,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def train_one_model(input_file: Path, output_dir: Path, args):
    """Train một classifier cho một file one-vs-other và lưu ra disk."""
    joblib, *_ = import_dependencies()

    texts, labels, target_category = load_binary_dataset(input_file)
    model = build_model(args)

    model.fit(texts, labels)

    model_path = output_dir / f"{input_file.stem}.joblib"
    joblib.dump(model, model_path)

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    return {
        "file": input_file.name,
        "model": str(model_path),
        "target_category": target_category,
        "positive": positive_count,
        "negative": negative_count,
        "total": len(labels),
    }


def train_all_models(input_dir: Path, output_dir: Path, args):
    """Train 14 binary classifiers từ toàn bộ file CSV trong data/one_vs_other."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {input_dir}")

    summary = []

    for input_file in csv_files:
        print(f"Training {input_file.name} ...")
        item = train_one_model(input_file, output_dir, args)
        summary.append(item)
        print(
            f"  saved={item['model']} "
            f"target={item['target_category']} "
            f"pos={item['positive']} neg={item['negative']}"
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "model_type": "TF-IDF + XGBoost",
                "text_input": "title_clean/title + description",
                "max_features_per_vectorizer": args.max_features,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "reg_lambda": args.reg_lambda,
                "tree_method": args.tree_method,
                "n_jobs": args.n_jobs,
                "seed": args.seed,
                "models": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def parse_args():
    """Đọc tham số dòng lệnh cho script train 14 binary classifiers."""
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + XGBoost cho 14 bộ one-vs-other."
    )
    parser.add_argument(
        "--input-dir",
        default=Path("data") / "one_vs_other",
        type=Path,
        help="Folder chứa 14 file CSV one-vs-other.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("models") / "one_vs_other_tfidf_xgboost",
        type=Path,
        help="Folder lưu các model .joblib.",
    )
    parser.add_argument(
        "--max-features",
        default=100000,
        type=int,
        help="Số feature tối đa cho mỗi TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--n-estimators",
        default=300,
        type=int,
        help="Số cây boosting.",
    )
    parser.add_argument(
        "--max-depth",
        default=6,
        type=int,
        help="Độ sâu tối đa của mỗi cây.",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.1,
        type=float,
        help="Learning rate của XGBoost.",
    )
    parser.add_argument(
        "--subsample",
        default=0.8,
        type=float,
        help="Tỷ lệ sample theo hàng cho mỗi cây.",
    )
    parser.add_argument(
        "--colsample-bytree",
        default=0.8,
        type=float,
        help="Tỷ lệ sample theo cột cho mỗi cây.",
    )
    parser.add_argument(
        "--reg-lambda",
        default=1.0,
        type=float,
        help="L2 regularization của XGBoost.",
    )
    parser.add_argument(
        "--tree-method",
        default="hist",
        help="Tree method của XGBoost. Gợi ý: hist.",
    )
    parser.add_argument(
        "--n-jobs",
        default=-1,
        type=int,
        help="Số CPU cores dùng để train.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed cho XGBoost.",
    )
    return parser.parse_args()


def main():
    """Hàm chính khi chạy trực tiếp file này từ dòng lệnh."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    summary = train_all_models(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        args=args,
    )
    print(f"Done. Trained {len(summary)} models in {args.output_dir}")


if __name__ == "__main__":
    main()
