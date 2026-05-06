import argparse
import csv
import json
import sys
from pathlib import Path

from train_one_vs_other_tflr_content import build_text_tc, build_text_tcd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_BUILDERS = {
    "tc": build_text_tc,
    "tcd": build_text_tcd,
}
VARIANT_DESCRIPTIONS = {
    "tc": "title + content",
    "tcd": "title + content + description",
}


def import_dependencies():
    try:
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ModuleNotFoundError as exc:
        missing_package = exc.name
        raise SystemExit(
            f"Missing package '{missing_package}'. "
            "Install with: pip install scikit-learn joblib"
        ) from exc

    return joblib, accuracy_score, precision_score, recall_score, f1_score


def resolve_model_dir(model_dir: Path) -> Path:
    if model_dir.exists():
        return model_dir

    fallback_dir = PROJECT_ROOT / "models" / "compare_tc_tcd"
    if model_dir != fallback_dir and fallback_dir.exists():
        print(f"Model dir not found: {model_dir}; using fallback {fallback_dir}.")
        return fallback_dir

    raise FileNotFoundError(f"Model dir not found: {model_dir}")


def load_rows_from_csv(csv_file: Path, required_columns):
    rows = []

    if not csv_file.exists():
        raise FileNotFoundError(f"Test file not found: {csv_file}")

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing_columns = set(required_columns) - set(fieldnames)
        if missing_columns:
            raise ValueError(
                f"File {csv_file} is missing columns: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Test file has no rows: {csv_file}")

    return rows


def load_category_test_files(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    csv_files = sorted(path for path in test_dir.glob("*.csv") if path.name != "summary.json")
    if not csv_files:
        raise ValueError(f"No test CSV files found in {test_dir}")

    rows_by_stem = {}
    for csv_file in csv_files:
        rows_by_stem[csv_file.stem] = load_rows_from_csv(
            csv_file=csv_file,
            required_columns=["binary_label", "target_category"],
        )

    return rows_by_stem


def parse_variant_from_model_stem(model_stem):
    for variant in VARIANT_BUILDERS:
        suffix = f"_{variant}"
        if model_stem.endswith(suffix):
            return model_stem[: -len(suffix)], variant

    raise ValueError(
        f"Cannot infer variant from model name '{model_stem}'. "
        "Expected suffix '_tc' or '_tcd'."
    )


def load_model_info_map(model_dir: Path):
    info_map = {}
    summary_path = model_dir / "summary.json"

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        for item in summary.get("models", []):
            model_path = item.get("model")
            target_category = item.get("target_category")
            csv_file = item.get("file")
            variant = item.get("variant")

            if not model_path or not target_category:
                continue

            model_stem = Path(model_path).stem
            test_stem = Path(csv_file).stem if csv_file else None

            if not variant or not test_stem:
                parsed_test_stem, parsed_variant = parse_variant_from_model_stem(model_stem)
                test_stem = test_stem or parsed_test_stem
                variant = variant or parsed_variant

            if variant not in VARIANT_BUILDERS:
                raise ValueError(f"Unsupported variant '{variant}' for model {model_stem}")

            info_map[model_stem] = {
                "target_category": target_category,
                "test_stem": test_stem,
                "variant": variant,
                "text_input": VARIANT_DESCRIPTIONS[variant],
            }

    return info_map


def get_model_info(model_path: Path, model_info_map):
    if model_path.stem in model_info_map:
        return model_info_map[model_path.stem]

    test_stem, variant = parse_variant_from_model_stem(model_path.stem)
    return {
        "target_category": None,
        "test_stem": test_stem,
        "variant": variant,
        "text_input": VARIANT_DESCRIPTIONS[variant],
    }


def load_category_map_from_test_summary(test_dir: Path):
    category_map = {}
    summary_path = test_dir / "summary.json"
    if not summary_path.exists():
        return category_map

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    for item in summary.get("files", []):
        file_name = item.get("file")
        category = item.get("target_category") or item.get("category")
        if file_name and category:
            category_map[Path(file_name).stem] = category

    return category_map


def evaluate_model(model_path, model_info, texts, y_true, target_test_file):
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
        "variant": model_info["variant"],
        "text_input": model_info["text_input"],
        "target_category": model_info["target_category"],
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_file",
        "variant",
        "text_input",
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


def summarize_by_variant(metrics):
    summary = {}
    for variant in sorted({item["variant"] for item in metrics}):
        rows = [item for item in metrics if item["variant"] == variant]
        summary[variant] = {
            "text_input": VARIANT_DESCRIPTIONS[variant],
            "model_count": len(rows),
            "mean_accuracy": sum(item["accuracy"] for item in rows) / len(rows),
            "mean_precision": sum(item["precision"] for item in rows) / len(rows),
            "mean_recall": sum(item["recall"] for item in rows) / len(rows),
            "mean_f1": sum(item["f1"] for item in rows) / len(rows),
        }
    return summary


def get_binary_test_set(rows_by_stem, target_stem):
    if target_stem not in rows_by_stem:
        available_files = ", ".join(f"{stem}.csv" for stem in sorted(rows_by_stem))
        raise ValueError(
            f"Cannot find test file for target '{target_stem}.csv'. "
            f"Available files: {available_files}"
        )

    return rows_by_stem[target_stem]


def evaluate_all_models(model_dir, test_dir, test_file, output_dir, label_column):
    model_dir = resolve_model_dir(model_dir)
    model_paths = sorted(model_dir.glob("*.joblib"))

    if not model_paths:
        raise ValueError(f"No .joblib model files found in {model_dir}")

    model_info_map = load_model_info_map(model_dir)
    category_map = load_category_map_from_test_summary(test_dir)

    rows_by_stem = None
    common_test_rows = None
    if test_file is not None:
        common_test_rows = load_rows_from_csv(
            csv_file=test_file,
            required_columns=[label_column],
        )
    else:
        rows_by_stem = load_category_test_files(test_dir)

    metrics = []
    for model_path in model_paths:
        model_info = get_model_info(model_path, model_info_map)
        if not model_info["target_category"]:
            model_info["target_category"] = category_map.get(model_info["test_stem"])

        if not model_info["target_category"]:
            raise ValueError(
                f"Cannot determine target category for model {model_path.name}. "
                "Check model summary.json or data/test/summary.json."
            )

        variant = model_info["variant"]
        text_builder = VARIANT_BUILDERS[variant]

        print(
            f"Evaluating {model_path.name}: "
            f"{model_info['target_category']} vs Other ({variant}) ..."
        )

        if common_test_rows is not None:
            test_rows = common_test_rows
            y_true = [
                1 if (row.get(label_column) or "").strip() == model_info["target_category"] else 0
                for row in test_rows
            ]
            target_test_file = str(test_file)
        else:
            test_rows = get_binary_test_set(
                rows_by_stem=rows_by_stem,
                target_stem=model_info["test_stem"],
            )
            y_true = [int(row["binary_label"]) for row in test_rows]
            target_test_file = str(test_dir / f"{model_info['test_stem']}.csv")

        texts = [text_builder(row) for row in test_rows]
        metrics.append(
            evaluate_model(
                model_path=model_path,
                model_info=model_info,
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
                "variants": VARIANT_DESCRIPTIONS,
                "aggregate_by_variant": summarize_by_variant(metrics),
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
    parser = argparse.ArgumentParser(
        description="Evaluate one-vs-other TF-IDF LR content variants: TC and TCD."
    )
    parser.add_argument(
        "--model-dir",
        default=PROJECT_ROOT / "models" / "compare_tc_tcd",
        type=Path,
        help="Folder containing *_tc.joblib and *_tcd.joblib models.",
    )
    parser.add_argument(
        "--test-dir",
        default=PROJECT_ROOT / "data" / "test",
        type=Path,
        help="Folder containing per-category one-vs-other test CSV files.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        type=Path,
        help="Optional common test file instead of per-category data/test files.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "data" / "reports_tflr_content",
        type=Path,
        help="Folder for evaluation reports.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Category label column in a common test file.",
    )
    return parser.parse_args()


def main():
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
