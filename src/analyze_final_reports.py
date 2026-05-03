import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORTS_DIR = PROJECT_ROOT / "data" / "final_reports"
OUTPUT_CSV = FINAL_REPORTS_DIR / "final_analyze.csv"


MODEL_LABELS = {
    "tfidf_lr": "TF_IDF + LR",
    "bert": "BERT",
    "tfidf_xgboost": "TF_IDF + XGBOOST",
}

METHOD_DIRS = {
    "max_voting": {
        "tfidf_lr": FINAL_REPORTS_DIR / "tfidf_lr_maxvoting" / "metrics.json",
        "bert": FINAL_REPORTS_DIR / "bert_maxvoting" / "metrics.json",
        "tfidf_xgboost": FINAL_REPORTS_DIR / "tfidf_xgboost_maxvoting" / "metrics.json",
    },
    "two_stage": {
        "tfidf_lr": FINAL_REPORTS_DIR / "tfidf_lr_stacking" / "metrics.json",
        "bert": FINAL_REPORTS_DIR / "bert_stacking" / "metrics.json",
        "tfidf_xgboost": FINAL_REPORTS_DIR / "tfidf_xgboost_stacking" / "metrics.json",
    },
}


def load_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def build_table_rows(metrics_by_method):
    max_threshold = metrics_by_method["max_voting"]["tfidf_lr"]["threshold"]
    stacking_threshold = metrics_by_method["two_stage"]["tfidf_lr"]["threshold"]

    rows = [
        ["", "", "Max voting", "", "", "Two-stage", "", ""],
        ["", "Metrics", f"Threshold = {max_threshold:.1f}", "", "", f"Threshold = {stacking_threshold:.1f}", "", ""],
        ["", "Model stage 1", "TF_IDF + LR", "BERT", "TF_IDF + XGBOOST", "TF_IDF + LR", "BERT", "TF_IDF + XGBOOST"],
        [
            "",
            "Accuracy",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["accuracy"]),
            fmt(metrics_by_method["max_voting"]["bert"]["accuracy"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["accuracy"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["accuracy"]),
            fmt(metrics_by_method["two_stage"]["bert"]["accuracy"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["accuracy"]),
        ],
        [
            "Micro",
            "Precision",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["micro_precision"]),
            fmt(metrics_by_method["max_voting"]["bert"]["micro_precision"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["micro_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["micro_precision"]),
            fmt(metrics_by_method["two_stage"]["bert"]["micro_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["micro_precision"]),
        ],
        [
            "",
            "Recall",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["micro_recall"]),
            fmt(metrics_by_method["max_voting"]["bert"]["micro_recall"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["micro_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["micro_recall"]),
            fmt(metrics_by_method["two_stage"]["bert"]["micro_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["micro_recall"]),
        ],
        [
            "",
            "F1",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["micro_f1"]),
            fmt(metrics_by_method["max_voting"]["bert"]["micro_f1"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["micro_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["micro_f1"]),
            fmt(metrics_by_method["two_stage"]["bert"]["micro_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["micro_f1"]),
        ],
        [
            "Macro",
            "Precision",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["macro_precision"]),
            fmt(metrics_by_method["max_voting"]["bert"]["macro_precision"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["macro_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["macro_precision"]),
            fmt(metrics_by_method["two_stage"]["bert"]["macro_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["macro_precision"]),
        ],
        [
            "",
            "Recall",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["macro_recall"]),
            fmt(metrics_by_method["max_voting"]["bert"]["macro_recall"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["macro_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["macro_recall"]),
            fmt(metrics_by_method["two_stage"]["bert"]["macro_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["macro_recall"]),
        ],
        [
            "",
            "F1",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["macro_f1"]),
            fmt(metrics_by_method["max_voting"]["bert"]["macro_f1"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["macro_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["macro_f1"]),
            fmt(metrics_by_method["two_stage"]["bert"]["macro_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["macro_f1"]),
        ],
        [
            "Weighted",
            "Precision",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["weighted_precision"]),
            fmt(metrics_by_method["max_voting"]["bert"]["weighted_precision"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["weighted_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["weighted_precision"]),
            fmt(metrics_by_method["two_stage"]["bert"]["weighted_precision"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["weighted_precision"]),
        ],
        [
            "",
            "Recall",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["weighted_recall"]),
            fmt(metrics_by_method["max_voting"]["bert"]["weighted_recall"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["weighted_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["weighted_recall"]),
            fmt(metrics_by_method["two_stage"]["bert"]["weighted_recall"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["weighted_recall"]),
        ],
        [
            "",
            "F1",
            fmt(metrics_by_method["max_voting"]["tfidf_lr"]["weighted_f1"]),
            fmt(metrics_by_method["max_voting"]["bert"]["weighted_f1"]),
            fmt(metrics_by_method["max_voting"]["tfidf_xgboost"]["weighted_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_lr"]["weighted_f1"]),
            fmt(metrics_by_method["two_stage"]["bert"]["weighted_f1"]),
            fmt(metrics_by_method["two_stage"]["tfidf_xgboost"]["weighted_f1"]),
        ],
    ]
    return rows


def main():
    metrics_by_method = {}
    for method_name, model_paths in METHOD_DIRS.items():
        metrics_by_method[method_name] = {
            model_name: load_metrics(path) for model_name, path in model_paths.items()
        }

    rows = build_table_rows(metrics_by_method)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
