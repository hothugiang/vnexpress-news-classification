import argparse
import csv
import json
import sys
from pathlib import Path


def import_sklearn_dependencies():
    """Import các thư viện sklearn/joblib và báo lỗi rõ nếu môi trường chưa cài."""
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ModuleNotFoundError as exc:
        missing_package = exc.name
        raise SystemExit(
            f"Thiếu thư viện '{missing_package}'. "
            "Cài bằng: pip install scikit-learn joblib"
        ) from exc

    return joblib, TfidfVectorizer, LogisticRegression, FeatureUnion, Pipeline


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


def build_model(max_features, max_iter, random_state):
    """Tạo pipeline TF-IDF + Logistic Regression cho một bộ binary."""
    (
        _joblib,
        TfidfVectorizer,
        LogisticRegression,
        FeatureUnion,
        Pipeline,
    ) = import_sklearn_dependencies()

    # Word n-gram học theo từ; char n-gram giúp mô hình ổn hơn với tiếng Việt.
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=max_iter,
        solver="liblinear",
        random_state=random_state,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def train_one_model(input_file: Path, output_dir: Path, args):
    """Train một classifier cho một file one-vs-other và lưu ra disk."""
    joblib, *_ = import_sklearn_dependencies()

    texts, labels, target_category = load_binary_dataset(input_file)
    model = build_model(
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

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
                "model_type": "TF-IDF + Logistic Regression",
                "text_input": "title_clean/title + description",
                "max_features_per_vectorizer": args.max_features,
                "max_iter": args.max_iter,
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
        description="Train TF-IDF + Logistic Regression cho 14 bộ one-vs-other."
    )
    parser.add_argument(
        "--input-dir",
        default=Path("data") / "one_vs_other",
        type=Path,
        help="Folder chứa 14 file CSV one-vs-other.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("models") / "one_vs_other",
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
        "--max-iter",
        default=1000,
        type=int,
        help="Số vòng lặp tối đa của Logistic Regression.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed cho Logistic Regression.",
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
