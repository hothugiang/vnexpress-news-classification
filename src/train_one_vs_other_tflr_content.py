import argparse
import csv
import json
import sys
import time
from pathlib import Path


def import_sklearn_dependencies():
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Thiếu thư viện '{exc.name}'. "
            "Cài bằng: pip install scikit-learn joblib"
        ) from exc

    return joblib, TfidfVectorizer, LogisticRegression, FeatureUnion, Pipeline


# ================= FEATURE BUILDERS =================

def build_text_tc(row):
    """title + content"""
    title = (row.get("title_clean") or row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    return f"{title} {content}".strip()


def build_text_tcd(row):
    """title + content + description"""
    title = (row.get("title_clean") or row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    description = (row.get("description") or "").strip()
    return f"{title} {content} {description}".strip()


# ================= DATA LOADING =================

def load_binary_dataset(input_file: Path, text_builder):
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
            texts.append(text_builder(row))
            labels.append(int(row["binary_label"]))
            target_category = row["target_category"]

    if target_category is None:
        raise ValueError(f"File {input_file} không có dữ liệu")

    return texts, labels, target_category


# ================= MODEL =================

def build_model(max_features, max_iter, random_state):
    (
        _joblib,
        TfidfVectorizer,
        LogisticRegression,
        FeatureUnion,
        Pipeline,
    ) = import_sklearn_dependencies()

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


# ================= TRAIN =================

def train_one_model_variant(input_file, output_dir, args, variant_name, text_builder):
    joblib, *_ = import_sklearn_dependencies()

    texts, labels, target_category = load_binary_dataset(input_file, text_builder)

    model = build_model(
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    # ⏱️ đo thời gian train
    start_time = time.time()
    model.fit(texts, labels)
    end_time = time.time()

    train_time = end_time - start_time

    model_path = output_dir / f"{input_file.stem}_{variant_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "file": input_file.name,
        "model": str(model_path),
        "target_category": target_category,
        "variant": variant_name,
        "train_time_sec": round(train_time, 4),
        "samples": len(labels),
    }


def train_all_models(input_dir: Path, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"Không tìm thấy CSV trong {input_dir}")

    summary = []
    total_start = time.time()

    for input_file in csv_files:
        print(f"\nTraining {input_file.name} ...")

        # TC
        item_tc = train_one_model_variant(
            input_file, output_dir, args,
            "tc", build_text_tc
        )
        summary.append(item_tc)
        print(f"  TC  -> {item_tc['train_time_sec']}s")

        # TCD
        item_tcd = train_one_model_variant(
            input_file, output_dir, args,
            "tcd", build_text_tcd
        )
        summary.append(item_tcd)
        print(f"  TCD -> {item_tcd['train_time_sec']}s")

    total_time = time.time() - total_start

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "model_type": "TF-IDF + Logistic Regression",
                "variants": {
                    "tc": "title + content",
                    "tcd": "title + content + description",
                },
                "total_models": len(summary),
                "total_train_time_sec": round(total_time, 2),
                "models": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDone. Total training time: {round(total_time, 2)}s")
    return summary


# ================= CLI =================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TC vs TCD models + log training time"
    )
    parser.add_argument(
        "--input-dir",
        default=Path("data") / "one_vs_other",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path("models") / "compare_tc_tcd",
        type=Path,
    )
    parser.add_argument("--max-features", default=100000, type=int)
    parser.add_argument("--max-iter", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    train_all_models(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        args=args,
    )


if __name__ == "__main__":
    main()