import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from evaluate_final_classifier_argmax import (
    build_text,
    import_common_dependencies,
    load_test_rows,
    load_tfidf_or_xgb_models,
    load_bert_model_dirs,
    score_bert,
    score_tfidf_or_xgb,
)


def import_stage2_dependencies():
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Thiếu thư viện '{exc.name}'. Cài bằng: pip install scikit-learn joblib"
        ) from exc

    return joblib, LogisticRegression


def get_stage1_scores(model_type: str, model_dir: Path, texts, bert_batch_size: int):
    """Sinh score 14 chiều từ 14 binary classifiers của stage 1."""
    if model_type in {"tfidf_lr", "tfidf_xgboost"}:
        models = load_tfidf_or_xgb_models(model_dir)
        categories = [item["target_category"] for item in models]
        score_by_category = score_tfidf_or_xgb(models, texts)
        return categories, score_by_category

    if model_type == "bert":
        model_items = load_bert_model_dirs(model_dir)
        categories = [item["target_category"] for item in model_items]
        score_by_category = score_bert(model_dir, texts, bert_batch_size)
        return categories, score_by_category

    raise ValueError(f"Model type không hợp lệ: {model_type}")


def build_feature_matrix(score_by_category, categories):
    """Chuyển dict category -> list(score) thành ma trận N x 14."""
    num_rows = len(next(iter(score_by_category.values())))
    features = []
    for row_idx in range(num_rows):
        features.append([float(score_by_category[category][row_idx]) for category in categories])
    return features


def write_feature_csv(output_path: Path, rows, categories, label_column):
    fieldnames = [f"score_{category}" for category in categories] + [label_column]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_csv(output_path: Path, rows, categories):
    fieldnames = [
        "true_category",
        "predicted_category",
        "predicted_score",
        "raw_best_category",
        "threshold_applied",
    ] + [f"proba_{category}" for category in categories]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_stacking(
    model_type: str,
    model_dir: Path,
    train_file: Path,
    test_file: Path,
    output_dir: Path,
    label_column: str,
    bert_batch_size: int,
    max_iter: int,
    seed: int,
    threshold: float,
    other_label: str,
):
    train_rows = load_test_rows(train_file, label_column)
    test_rows = load_test_rows(test_file, label_column)

    train_texts = [build_text(row) for row in train_rows]
    test_texts = [build_text(row) for row in test_rows]
    y_train = [(row.get(label_column) or "").strip() for row in train_rows]
    y_test = [(row.get(label_column) or "").strip() for row in test_rows]

    print("Scoring stage 1 models on train split ...")
    categories, train_scores = get_stage1_scores(
        model_type=model_type,
        model_dir=model_dir,
        texts=train_texts,
        bert_batch_size=bert_batch_size,
    )

    print("Scoring stage 1 models on test split ...")
    _, test_scores = get_stage1_scores(
        model_type=model_type,
        model_dir=model_dir,
        texts=test_texts,
        bert_batch_size=bert_batch_size,
    )

    x_train = build_feature_matrix(train_scores, categories)
    x_test = build_feature_matrix(test_scores, categories)

    joblib, LogisticRegression = import_stage2_dependencies()
    stage2_model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    stage2_model.fit(x_train, y_train)

    y_pred_proba = stage2_model.predict_proba(x_test)
    stage2_classes = list(stage2_model.classes_)
    y_pred = []
    threshold_fallback_count = 0
    for proba_row in y_pred_proba:
        best_idx = max(range(len(proba_row)), key=lambda idx: proba_row[idx])
        best_score = float(proba_row[best_idx])
        best_category = stage2_classes[best_idx]
        final_category = best_category if best_score >= threshold else other_label
        if final_category != best_category:
            threshold_fallback_count += 1
        y_pred.append(final_category)
    evaluation_labels = list(dict.fromkeys(stage2_classes + [other_label] + sorted(set(y_test))))

    (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    ) = import_common_dependencies()

    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_model_path = output_dir / "stage2_logistic_regression.joblib"
    joblib.dump(stage2_model, stage2_model_path)

    train_feature_rows = []
    for features, label in zip(x_train, y_train):
        row = {f"score_{category}": value for category, value in zip(categories, features)}
        row[label_column] = label
        train_feature_rows.append(row)
    write_feature_csv(output_dir / "train_stage2_features.csv", train_feature_rows, categories, label_column)

    test_feature_rows = []
    for features, label in zip(x_test, y_test):
        row = {f"score_{category}": value for category, value in zip(categories, features)}
        row[label_column] = label
        test_feature_rows.append(row)
    write_feature_csv(output_dir / "test_stage2_features.csv", test_feature_rows, categories, label_column)

    prediction_rows = []
    for true_category, predicted_category, proba_row in zip(y_test, y_pred, y_pred_proba):
        class_prob_map = {category: 0.0 for category in stage2_classes}
        class_prob_map.update(
            {category: float(prob) for category, prob in zip(stage2_classes, proba_row)}
        )
        raw_best_category = max(stage2_classes, key=lambda category: class_prob_map[category])
        best_score = class_prob_map[raw_best_category]
        row = {
            "true_category": true_category,
            "predicted_category": predicted_category,
            "predicted_score": best_score,
            "raw_best_category": raw_best_category,
            "threshold_applied": int(predicted_category != raw_best_category),
        }
        for category in stage2_classes:
            row[f"proba_{category}"] = class_prob_map[category]
        prediction_rows.append(row)
    predictions_csv = output_dir / "predictions.csv"
    write_predictions_csv(predictions_csv, prediction_rows, stage2_classes)

    class_report_dict = classification_report(
        y_test,
        y_pred,
        labels=evaluation_labels,
        output_dict=True,
        zero_division=0,
    )
    class_report_text = classification_report(
        y_test,
        y_pred,
        labels=evaluation_labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=evaluation_labels)
    confusion_csv = output_dir / "confusion_matrix.csv"
    with confusion_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + evaluation_labels)
        for category, row in zip(evaluation_labels, cm.tolist()):
            writer.writerow([category] + row)

    metrics = {
        "model_type": model_type,
        "stage_2_method": "stacking_logistic_regression_threshold_no_oof",
        "train_file": str(train_file),
        "test_file": str(test_file),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "stage2_num_features": len(categories),
        "stage2_classes": stage2_classes,
        "threshold": threshold,
        "other_label": other_label,
        "threshold_fallback_count": threshold_fallback_count,
        "accuracy": accuracy_score(y_test, y_pred),
        "micro_precision": precision_score(y_test, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_test, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
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
                "stage_2_method": "stacking_logistic_regression_threshold_no_oof",
                "note": (
                    "Stage 2 features are created directly from stage 1 models "
                    "trained on the full train split; OOF stacking is not used."
                ),
                "model_dir": str(model_dir),
                "train_file": str(train_file),
                "test_file": str(test_file),
                "threshold": threshold,
                "other_label": other_label,
                "stage2_model_path": str(stage2_model_path),
                "outputs": {
                    "train_features_csv": str(output_dir / "train_stage2_features.csv"),
                    "test_features_csv": str(output_dir / "test_stage2_features.csv"),
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

    print(f"Done. Saved final stacking evaluation to: {output_dir}")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final multiclass stage 2 stacking classifier using "
            "14 stage 1 binary outputs as features."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["tfidf_lr", "tfidf_xgboost", "bert"],
        required=True,
        help="Họ model của 14 binary classifiers ở stage 1.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Thư mục model stage 1. Nếu bỏ trống sẽ dùng default theo model-type.",
    )
    parser.add_argument(
        "--train-file",
        default=PROJECT_ROOT / "data" / "split" / "train.csv",
        type=Path,
        help="Train split gốc đa lớp.",
    )
    parser.add_argument(
        "--test-file",
        default=PROJECT_ROOT / "data" / "split" / "test.csv",
        type=Path,
        help="Test split gốc đa lớp.",
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
        help="Tên cột nhãn đa lớp trong train/test split.",
    )
    parser.add_argument(
        "--bert-batch-size",
        default=64,
        type=int,
        help="Batch size cho BERT khi sinh score stage 1.",
    )
    parser.add_argument(
        "--max-iter",
        default=1000,
        type=int,
        help="Số vòng lặp tối đa cho stage 2 Logistic Regression.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed cho stage 2 Logistic Regression.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Nếu xác suất cao nhất của stage 2 < threshold thì gán về nhãn Khác.",
    )
    parser.add_argument(
        "--other-label",
        default="Khác",
        help="Nhãn fallback khi không class nào vượt threshold.",
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
        "tfidf_lr": PROJECT_ROOT / "data" / "final_reports" / "tfidf_lr_stacking",
        "tfidf_xgboost": PROJECT_ROOT / "data" / "final_reports" / "tfidf_xgboost_stacking",
        "bert": PROJECT_ROOT / "data" / "final_reports" / "bert_stacking",
    }

    model_dir = args.model_dir or default_model_dirs[args.model_type]
    output_dir = args.output_dir or default_output_dirs[args.model_type]

    evaluate_stacking(
        model_type=args.model_type,
        model_dir=model_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=output_dir,
        label_column=args.label_column,
        bert_batch_size=args.bert_batch_size,
        max_iter=args.max_iter,
        seed=args.seed,
        threshold=args.threshold,
        other_label=args.other_label,
    )


if __name__ == "__main__":
    main()
