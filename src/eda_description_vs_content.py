import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "EDA kiểm chứng tương quan nội dung giữa description/content và "
            "so sánh khả năng dự đoán category khi chỉ dùng description."
        )
    )
    parser.add_argument(
        "--input",
        default=Path("data_cleaned.csv"),
        type=Path,
        help="File CSV đã clean.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data") / "eda_description_vs_content",
        type=Path,
        help="Thư mục lưu output EDA.",
    )
    parser.add_argument(
        "--description-column",
        default="description",
        help="Tên cột mô tả ngắn.",
    )
    parser.add_argument(
        "--content-column",
        default="content",
        help="Tên cột nội dung bài viết.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Tên cột category cần dự đoán.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed để kết quả tái lập được.",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Tỷ lệ test set cho thí nghiệm classifier.",
    )
    parser.add_argument(
        "--similarity-sample-size",
        default=30000,
        type=int,
        help=(
            "Số dòng lấy mẫu stratified để đo similarity. "
            "Đặt 0 hoặc số >= dataset size để dùng toàn bộ dữ liệu."
        ),
    )
    parser.add_argument(
        "--max-features-similarity",
        default=80000,
        type=int,
        help="Số feature TF-IDF tối đa cho phần similarity.",
    )
    parser.add_argument(
        "--max-features-model",
        default=120000,
        type=int,
        help="Số feature TF-IDF tối đa cho mỗi classifier.",
    )
    parser.add_argument(
        "--min-df",
        default=2,
        type=int,
        help="min_df cho TF-IDF.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Chỉ chạy EDA word count/similarity, bỏ qua classifier.",
    )
    return parser.parse_args()


def read_dataset(args):
    required_columns = [
        args.description_column,
        args.content_column,
        args.label_column,
    ]

    df = pd.read_csv(args.input, usecols=required_columns, dtype=str)
    initial_rows = len(df)

    for column in required_columns:
        df[column] = df[column].fillna("").astype(str)

    non_empty_mask = (
        (df[args.description_column].str.strip() != "")
        & (df[args.content_column].str.strip() != "")
        & (df[args.label_column].str.strip() != "")
    )
    df = df.loc[non_empty_mask].copy()

    return df, {
        "input_file": str(args.input),
        "initial_rows": int(initial_rows),
        "rows_after_drop_empty_required_fields": int(len(df)),
        "dropped_rows": int(initial_rows - len(df)),
        "description_column": args.description_column,
        "content_column": args.content_column,
        "label_column": args.label_column,
    }


def word_count(text):
    return len(str(text).split())


def describe_numeric(series):
    quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p50_median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(series.max()),
    }


def add_word_count_features(df, args):
    df = df.copy()
    df["description_word_count"] = df[args.description_column].map(word_count)
    df["content_word_count"] = df[args.content_column].map(word_count)
    df["description_to_content_word_ratio"] = (
        df["description_word_count"] / df["content_word_count"].replace(0, np.nan)
    )
    return df


def build_word_count_outputs(df, args):
    summary_rows = []
    for feature in ["description_word_count", "content_word_count"]:
        item = describe_numeric(df[feature])
        item["feature"] = feature
        summary_rows.append(item)

    ratio_summary = describe_numeric(df["description_to_content_word_ratio"].dropna())
    ratio_summary["feature"] = "description_to_content_word_ratio"
    summary_rows.append(ratio_summary)

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = ["feature"] + [column for column in summary_df.columns if column != "feature"]
    summary_df = summary_df[ordered_columns]

    by_category = (
        df.groupby(args.label_column)
        .agg(
            rows=(args.label_column, "size"),
            description_word_mean=("description_word_count", "mean"),
            description_word_median=("description_word_count", "median"),
            content_word_mean=("content_word_count", "mean"),
            content_word_median=("content_word_count", "median"),
            ratio_median=("description_to_content_word_ratio", "median"),
            ratio_p90=("description_to_content_word_ratio", lambda s: s.quantile(0.9)),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    distribution = (
        df[args.label_column]
        .value_counts()
        .rename_axis(args.label_column)
        .reset_index(name="rows")
    )
    distribution["share"] = distribution["rows"] / len(df)

    aggregate = {
        "description_mean_words": float(df["description_word_count"].mean()),
        "description_median_words": float(df["description_word_count"].median()),
        "content_mean_words": float(df["content_word_count"].mean()),
        "content_median_words": float(df["content_word_count"].median()),
        "mean_description_words_div_mean_content_words": float(
            df["description_word_count"].mean() / df["content_word_count"].mean()
        ),
        "median_description_words_div_median_content_words": float(
            df["description_word_count"].median()
            / df["content_word_count"].median()
        ),
        "row_ratio_median": float(df["description_to_content_word_ratio"].median()),
        "row_ratio_p90": float(df["description_to_content_word_ratio"].quantile(0.9)),
        "rows_content_shorter_than_description": int(
            (df["content_word_count"] < df["description_word_count"]).sum()
        ),
        "rows_content_under_50_words": int((df["content_word_count"] < 50).sum()),
        "rows_content_under_100_words": int((df["content_word_count"] < 100).sum()),
    }

    return summary_df, by_category, distribution, aggregate


def stratified_sample(df, label_column, sample_size, seed):
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    sample_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[label_column],
        random_state=seed,
    )
    return sample_df.reset_index(drop=True)


def token_set(text):
    return set(TOKEN_PATTERN.findall(str(text).lower()))


def compute_overlap_metrics(description, content):
    description_tokens = token_set(description)
    content_tokens = token_set(content)

    if not description_tokens and not content_tokens:
        return 0.0, 0.0, 0.0

    intersection_size = len(description_tokens & content_tokens)
    union_size = max(len(description_tokens | content_tokens), 1)

    jaccard = intersection_size / union_size
    description_coverage = intersection_size / max(len(description_tokens), 1)
    content_coverage = intersection_size / max(len(content_tokens), 1)
    return jaccard, description_coverage, content_coverage


def compute_similarity(df, args):
    sample_df = stratified_sample(
        df=df,
        label_column=args.label_column,
        sample_size=args.similarity_sample_size,
        seed=args.seed,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features_similarity,
        sublinear_tf=True,
        norm="l2",
    )
    fit_texts = pd.concat(
        [sample_df[args.description_column], sample_df[args.content_column]],
        ignore_index=True,
    )
    vectorizer.fit(fit_texts)

    description_matrix = vectorizer.transform(sample_df[args.description_column])
    content_matrix = vectorizer.transform(sample_df[args.content_column])

    same_row_cosine = np.asarray(
        description_matrix.multiply(content_matrix).sum(axis=1)
    ).ravel()

    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(content_matrix.shape[0])
    shuffled_cosine = np.asarray(
        description_matrix.multiply(content_matrix[shuffled_indices]).sum(axis=1)
    ).ravel()

    overlap_values = np.array(
        [
            compute_overlap_metrics(row.description, row.content)
            for row in sample_df.rename(
                columns={
                    args.description_column: "description",
                    args.content_column: "content",
                }
            )[["description", "content"]].itertuples(index=False)
        ]
    )

    sample_df = sample_df.copy()
    sample_df["tfidf_cosine_same_row"] = same_row_cosine
    sample_df["tfidf_cosine_shuffled_content"] = shuffled_cosine
    sample_df["token_jaccard"] = overlap_values[:, 0]
    sample_df["description_token_coverage_in_content"] = overlap_values[:, 1]
    sample_df["content_token_coverage_in_description"] = overlap_values[:, 2]

    summary = {
        "sample_rows": int(len(sample_df)),
        "tfidf_features": int(len(vectorizer.get_feature_names_out())),
        "tfidf_cosine_same_row": describe_numeric(sample_df["tfidf_cosine_same_row"]),
        "tfidf_cosine_shuffled_content": describe_numeric(
            sample_df["tfidf_cosine_shuffled_content"]
        ),
        "tfidf_cosine_mean_lift_vs_shuffled": float(
            sample_df["tfidf_cosine_same_row"].mean()
            / max(sample_df["tfidf_cosine_shuffled_content"].mean(), 1e-12)
        ),
        "token_jaccard": describe_numeric(sample_df["token_jaccard"]),
        "description_token_coverage_in_content": describe_numeric(
            sample_df["description_token_coverage_in_content"]
        ),
        "content_token_coverage_in_description": describe_numeric(
            sample_df["content_token_coverage_in_description"]
        ),
    }

    by_category = (
        sample_df.groupby(args.label_column)
        .agg(
            rows=(args.label_column, "size"),
            tfidf_cosine_same_row_mean=("tfidf_cosine_same_row", "mean"),
            tfidf_cosine_shuffled_mean=("tfidf_cosine_shuffled_content", "mean"),
            token_jaccard_mean=("token_jaccard", "mean"),
            description_token_coverage_mean=(
                "description_token_coverage_in_content",
                "mean",
            ),
            content_token_coverage_mean=(
                "content_token_coverage_in_description",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_category["tfidf_cosine_lift_vs_shuffled"] = (
        by_category["tfidf_cosine_same_row_mean"]
        / by_category["tfidf_cosine_shuffled_mean"].replace(0, np.nan)
    )

    return summary, by_category, sample_df


def build_text_feature(train_df, test_df, args, feature_name):
    if feature_name == "description":
        return train_df[args.description_column], test_df[args.description_column]
    if feature_name == "content":
        return train_df[args.content_column], test_df[args.content_column]
    if feature_name == "description_plus_content":
        return (
            train_df[args.description_column].str.cat(train_df[args.content_column], sep=" "),
            test_df[args.description_column].str.cat(test_df[args.content_column], sep=" "),
        )
    raise ValueError(f"Unsupported feature_name: {feature_name}")


def evaluate_text_feature(train_df, test_df, args, feature_name):
    train_text, test_text = build_text_feature(train_df, test_df, args, feature_name)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                    max_features=args.max_features_model,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(random_state=args.seed)),
        ]
    )

    model.fit(train_text, train_df[args.label_column])
    predictions = model.predict(test_text)

    metrics = {
        "text_feature": feature_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df[args.label_column], predictions)),
        "macro_precision": float(
            precision_score(
                test_df[args.label_column],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                test_df[args.label_column],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(test_df[args.label_column], predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(test_df[args.label_column], predictions, average="weighted")
        ),
    }

    report = classification_report(
        test_df[args.label_column],
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return metrics, report


def compare_classifiers(df, args):
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[args.label_column],
        random_state=args.seed,
    )

    metrics = []
    reports = {}
    for feature_name in ["description", "content", "description_plus_content"]:
        item, report = evaluate_text_feature(train_df, test_df, args, feature_name)
        metrics.append(item)
        reports[feature_name] = report

    metrics_df = pd.DataFrame(metrics)
    metrics_by_feature = metrics_df.set_index("text_feature").to_dict(orient="index")

    description = metrics_by_feature["description"]
    content = metrics_by_feature["content"]
    combined = metrics_by_feature["description_plus_content"]
    comparison = {
        "description_accuracy_minus_content_accuracy": float(
            description["accuracy"] - content["accuracy"]
        ),
        "description_accuracy_minus_combined_accuracy": float(
            description["accuracy"] - combined["accuracy"]
        ),
        "description_macro_f1_minus_content_macro_f1": float(
            description["macro_f1"] - content["macro_f1"]
        ),
        "description_weighted_f1_retention_vs_content": float(
            description["weighted_f1"] / max(content["weighted_f1"], 1e-12)
        ),
        "description_weighted_f1_retention_vs_combined": float(
            description["weighted_f1"] / max(combined["weighted_f1"], 1e-12)
        ),
    }

    return metrics_df, reports, comparison


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def number(value, digits=3):
    return f"{value:.{digits}f}"


def make_conclusion(data_profile, word_aggregate, similarity_summary, model_comparison):
    same_cosine = similarity_summary["tfidf_cosine_same_row"]["mean"]
    shuffled_cosine = similarity_summary["tfidf_cosine_shuffled_content"]["mean"]
    cosine_lift = similarity_summary["tfidf_cosine_mean_lift_vs_shuffled"]
    description_coverage = similarity_summary[
        "description_token_coverage_in_content"
    ]["mean"]

    if model_comparison is None:
        return {
            "verdict": "not_evaluated",
            "message": "Chưa chạy classifier vì dùng --skip-model.",
        }

    classifier_rows = model_comparison["classifier_metrics"]
    by_feature = {
        row["text_feature"]: row
        for row in classifier_rows
    }
    description = by_feature["description"]
    content = by_feature["content"]
    combined = by_feature["description_plus_content"]
    retention_vs_content = model_comparison[
        "description_weighted_f1_retention_vs_content"
    ]
    retention_vs_combined = model_comparison[
        "description_weighted_f1_retention_vs_combined"
    ]

    # Tiêu chí EDA thực dụng: description được xem là đủ nếu đạt accuracy tốt,
    # giữ lại >= 90% weighted F1 so với content/combined, và similarity cùng dòng
    # cao rõ rệt hơn baseline tráo content.
    sufficient = (
        description["accuracy"] >= 0.85
        and retention_vs_content >= 0.90
        and retention_vs_combined >= 0.90
        and cosine_lift >= 5.0
        and description_coverage >= 0.65
    )

    if sufficient:
        verdict = "support_description_only"
        message = (
            "Các metric ủng hộ việc chỉ dùng description khi ưu tiên mô hình gọn, "
            "dự đoán nhanh và chấp nhận mất một phần hiệu năng so với content."
        )
    else:
        verdict = "do_not_drop_content_without_more_validation"
        message = (
            "Các metric chưa đủ mạnh để bỏ content nếu mục tiêu là tối đa hóa hiệu năng."
        )

    return {
        "verdict": verdict,
        "message": message,
        "criteria": {
            "description_accuracy_min": 0.85,
            "weighted_f1_retention_min": 0.90,
            "cosine_lift_min": 5.0,
            "description_token_coverage_min": 0.65,
        },
        "observed": {
            "rows": data_profile["rows_after_drop_empty_required_fields"],
            "description_accuracy": description["accuracy"],
            "description_macro_f1": description["macro_f1"],
            "description_weighted_f1": description["weighted_f1"],
            "content_accuracy": content["accuracy"],
            "content_macro_f1": content["macro_f1"],
            "content_weighted_f1": content["weighted_f1"],
            "combined_accuracy": combined["accuracy"],
            "combined_macro_f1": combined["macro_f1"],
            "combined_weighted_f1": combined["weighted_f1"],
            "description_weighted_f1_retention_vs_content": retention_vs_content,
            "description_weighted_f1_retention_vs_combined": retention_vs_combined,
            "same_row_tfidf_cosine_mean": same_cosine,
            "shuffled_tfidf_cosine_mean": shuffled_cosine,
            "same_row_tfidf_cosine_lift_vs_shuffled": cosine_lift,
            "description_token_coverage_in_content_mean": description_coverage,
            "description_median_words": word_aggregate["description_median_words"],
            "content_median_words": word_aggregate["content_median_words"],
            "median_description_words_div_median_content_words": word_aggregate[
                "median_description_words_div_median_content_words"
            ],
        },
    }


def write_report(
    output_dir,
    data_profile,
    word_aggregate,
    similarity_summary,
    model_comparison,
    conclusion,
):
    same_cosine = similarity_summary["tfidf_cosine_same_row"]["mean"]
    same_median = similarity_summary["tfidf_cosine_same_row"]["p50_median"]
    shuffled_cosine = similarity_summary["tfidf_cosine_shuffled_content"]["mean"]
    shuffled_median = similarity_summary["tfidf_cosine_shuffled_content"]["p50_median"]
    cosine_lift = similarity_summary["tfidf_cosine_mean_lift_vs_shuffled"]
    description_coverage = similarity_summary[
        "description_token_coverage_in_content"
    ]["mean"]
    jaccard = similarity_summary["token_jaccard"]["mean"]

    lines = [
        "# EDA description vs content",
        "",
        "## Mục tiêu",
        "",
        (
            "File này kiểm chứng liệu `description` có đủ đại diện cho `content` "
            "để dùng làm feature chính dự đoán `category/main_category`, đồng thời "
            "thống kê và so sánh số từ giữa hai trường văn bản."
        ),
        "",
        "## Dữ liệu",
        "",
        f"- Input: `{data_profile['input_file']}`",
        f"- Số dòng ban đầu: {data_profile['initial_rows']:,}",
        (
            "- Số dòng dùng cho EDA sau khi bỏ rỗng "
            f"`{data_profile['description_column']}`, "
            f"`{data_profile['content_column']}`, "
            f"`{data_profile['label_column']}`: "
            f"{data_profile['rows_after_drop_empty_required_fields']:,}"
        ),
        "",
        "## So sánh số từ",
        "",
        (
            f"- `description`: trung bình {word_aggregate['description_mean_words']:.1f} từ, "
            f"median {word_aggregate['description_median_words']:.0f} từ."
        ),
        (
            f"- `content`: trung bình {word_aggregate['content_mean_words']:.1f} từ, "
            f"median {word_aggregate['content_median_words']:.0f} từ."
        ),
        (
            "- Theo median, `description` chỉ dài bằng "
            f"{pct(word_aggregate['median_description_words_div_median_content_words'])} "
            "so với `content`."
        ),
        (
            "- Có "
            f"{word_aggregate['rows_content_shorter_than_description']:,} dòng "
            "content ngắn hơn description; đây là nhóm nên kiểm tra thêm nếu cần làm sạch dữ liệu."
        ),
        "",
        "## Tương quan nội dung description-content",
        "",
        (
            "- TF-IDF cosine cùng dòng: "
            f"mean {number(same_cosine)}, median {number(same_median)}."
        ),
        (
            "- Baseline tráo `content` sang dòng khác: "
            f"mean {number(shuffled_cosine)}, median {number(shuffled_median)}."
        ),
        (
            "- Cùng dòng cao hơn baseline tráo dòng "
            f"{number(cosine_lift, 2)} lần theo mean cosine."
        ),
        (
            "- Trung bình "
            f"{pct(description_coverage)} token riêng biệt của `description` "
            "cũng xuất hiện trong `content`."
        ),
        (
            "- Jaccard token trung bình là "
            f"{number(jaccard)}; chỉ số này thấp hơn coverage vì `content` dài hơn rất nhiều."
        ),
        "",
    ]

    if model_comparison is not None:
        by_feature = {
            row["text_feature"]: row
            for row in model_comparison["classifier_metrics"]
        }
        desc = by_feature["description"]
        content = by_feature["content"]
        combined = by_feature["description_plus_content"]
        lines.extend(
            [
                "## Kiểm chứng bằng classifier",
                "",
                (
                    "- `description` only: "
                    f"accuracy {pct(desc['accuracy'])}, "
                    f"macro F1 {pct(desc['macro_f1'])}, "
                    f"weighted F1 {pct(desc['weighted_f1'])}."
                ),
                (
                    "- `content` only: "
                    f"accuracy {pct(content['accuracy'])}, "
                    f"macro F1 {pct(content['macro_f1'])}, "
                    f"weighted F1 {pct(content['weighted_f1'])}."
                ),
                (
                    "- `description + content`: "
                    f"accuracy {pct(combined['accuracy'])}, "
                    f"macro F1 {pct(combined['macro_f1'])}, "
                    f"weighted F1 {pct(combined['weighted_f1'])}."
                ),
                (
                    "- `description` giữ lại "
                    f"{pct(model_comparison['description_weighted_f1_retention_vs_content'])} "
                    "weighted F1 so với `content` only và "
                    f"{pct(model_comparison['description_weighted_f1_retention_vs_combined'])} "
                    "so với dùng cả hai trường."
                ),
                "",
                "## Kết luận",
                "",
                f"- Verdict: `{conclusion['verdict']}`.",
                f"- {conclusion['message']}",
                (
                    "- Diễn giải thực dụng: có thể chỉ dùng `description` nếu mục tiêu là "
                    "feature ngắn, train/predict nhanh và chấp nhận giảm khoảng "
                    f"{pct(content['accuracy'] - desc['accuracy'])} accuracy point so với `content`. "
                    "Nếu mục tiêu là tối đa hóa hiệu năng, `content` vẫn có giá trị bổ sung."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "## Kết luận",
                "",
                "- Chưa chạy phần classifier vì bật `--skip-model`.",
                f"- {conclusion['message']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `category_distribution.csv`",
            "- `word_count_summary.csv`",
            "- `word_count_by_category.csv`",
            "- `text_similarity_summary.json`",
            "- `text_similarity_by_category.csv`",
            "- `text_similarity_sample.csv`",
            "- `classifier_feature_comparison.csv`",
            "- `classification_report_by_feature.json`",
            "- `similarity_mean_same_vs_shuffled.png`",
            "- `similarity_key_metrics_bar.png`",
            "- `description_token_coverage_distribution.png`",
            "- `similarity_by_category.png`",
            "- `description_token_coverage_by_category.png`",
            "- `eda_summary.json`",
            "",
        ]
    )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_similarity_explanation_plots(output_dir, similarity_sample, label_column, plt):
    plot_paths = []

    same_mean = similarity_sample["tfidf_cosine_same_row"].mean()
    shuffled_mean = similarity_sample["tfidf_cosine_shuffled_content"].mean()
    lift = same_mean / max(shuffled_mean, 1e-12)
    coverage_mean = similarity_sample["description_token_coverage_in_content"].mean()
    jaccard_mean = similarity_sample["token_jaccard"].mean()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(
        ["Same row", "Shuffled baseline"],
        [same_mean, shuffled_mean],
        color=["#0f766e", "#a16207"],
    )
    ax.set_ylim(0, max(0.26, same_mean * 1.28))
    ax.set_title("Mean TF-IDF cosine: description vs content")
    ax.set_ylabel("Cosine similarity")
    ax.bar_label(bars, labels=[f"{same_mean:.3f}", f"{shuffled_mean:.3f}"], padding=4)
    ax.text(
        0.5,
        ax.get_ylim()[1] * 0.86,
        f"Same-row lift: {lift:.2f}x",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / "similarity_mean_same_vs_shuffled.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        [
            "Same-row\ncosine",
            "Shuffled\nbaseline",
            "Description token\ncoverage",
            "Token\nJaccard",
        ],
        [same_mean, shuffled_mean, coverage_mean, jaccard_mean],
        color=["#0f766e", "#a16207", "#2563eb", "#7c3aed"],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Key similarity metrics")
    ax.set_ylabel("Score")
    ax.bar_label(
        bars,
        labels=[
            f"{same_mean:.3f}",
            f"{shuffled_mean:.3f}",
            f"{coverage_mean:.1%}",
            f"{jaccard_mean:.3f}",
        ],
        padding=4,
    )
    fig.tight_layout()
    path = output_dir / "similarity_key_metrics_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(
        similarity_sample["description_token_coverage_in_content"],
        bins=np.linspace(0, 1, 31),
        color="#2563eb",
        alpha=0.85,
    )
    ax.axvline(
        coverage_mean,
        color="#b91c1c",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {coverage_mean:.1%}",
    )
    ax.set_xlim(0, 1)
    ax.set_title("Description token coverage in content")
    ax.set_xlabel("Share of unique description tokens also found in content")
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "description_token_coverage_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(path))

    if label_column in similarity_sample.columns:
        category_df = (
            similarity_sample.groupby(label_column)
            .agg(
                same_row_cosine=("tfidf_cosine_same_row", "mean"),
                shuffled_cosine=("tfidf_cosine_shuffled_content", "mean"),
                description_coverage=(
                    "description_token_coverage_in_content",
                    "mean",
                ),
            )
            .sort_values("same_row_cosine")
        )

        fig, ax = plt.subplots(figsize=(9, 6.2))
        y_pos = np.arange(len(category_df))
        ax.barh(
            y_pos - 0.18,
            category_df["same_row_cosine"],
            height=0.36,
            label="Same row",
            color="#0f766e",
        )
        ax.barh(
            y_pos + 0.18,
            category_df["shuffled_cosine"],
            height=0.36,
            label="Shuffled baseline",
            color="#a16207",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(category_df.index)
        ax.set_xlabel("Mean TF-IDF cosine")
        ax.set_title("TF-IDF cosine by category")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = output_dir / "similarity_by_category.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(9, 6.2))
        coverage_df = category_df.sort_values("description_coverage")
        bars = ax.barh(
            coverage_df.index,
            coverage_df["description_coverage"],
            color="#2563eb",
            alpha=0.9,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean coverage")
        ax.set_title("Description token coverage by category")
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in coverage_df["description_coverage"]], padding=4)
        fig.tight_layout()
        path = output_dir / "description_token_coverage_by_category.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

    return plot_paths


def try_write_plots(output_dir, df, similarity_sample, classifier_metrics, label_column):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plots because matplotlib is not usable: {exc}")
        return []

    plot_paths = []

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].hist(df["description_word_count"], bins=30, color="#2563eb", alpha=0.85)
        axes[0].set_title("Description word count")
        axes[0].set_xlabel("Words")
        axes[0].set_ylabel("Rows")

        positive_content = df.loc[df["content_word_count"] > 0, "content_word_count"]
        bins = np.logspace(
            np.log10(max(1, positive_content.min())),
            np.log10(positive_content.max()),
            45,
        )
        axes[1].hist(positive_content, bins=bins, color="#c2410c", alpha=0.85)
        axes[1].set_xscale("log")
        axes[1].set_title("Content word count")
        axes[1].set_xlabel("Words, log scale")
        axes[1].set_ylabel("Rows")
        fig.tight_layout()
        path = output_dir / "word_count_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(
            similarity_sample["tfidf_cosine_same_row"],
            bins=45,
            alpha=0.75,
            label="Same row",
            color="#0f766e",
        )
        ax.hist(
            similarity_sample["tfidf_cosine_shuffled_content"],
            bins=45,
            alpha=0.65,
            label="Shuffled content baseline",
            color="#a16207",
        )
        ax.set_title("TF-IDF cosine: description vs content")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Rows")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "similarity_same_vs_shuffled.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

        plot_paths.extend(
            write_similarity_explanation_plots(
                output_dir=output_dir,
                similarity_sample=similarity_sample,
                label_column=label_column,
                plt=plt,
            )
        )

        if classifier_metrics is not None and not classifier_metrics.empty:
            metric_cols = ["accuracy", "macro_f1", "weighted_f1"]
            plot_df = classifier_metrics.set_index("text_feature")[metric_cols]
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            plot_df.plot(kind="bar", ax=ax, color=["#2563eb", "#16a34a", "#c2410c"])
            ax.set_ylim(0, 1)
            ax.set_title("Classifier metrics by text feature")
            ax.set_xlabel("")
            ax.set_ylabel("Score")
            ax.legend(loc="lower right")
            fig.tight_layout()
            path = output_dir / "classifier_feature_comparison.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plot_paths.append(str(path))
    except Exception as exc:
        print(f"Skip plots because rendering failed: {exc}")

    return plot_paths


def main():
    configure_stdout()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input} ...")
    df, data_profile = read_dataset(args)
    df = add_word_count_features(df, args)

    print("Computing word-count statistics ...")
    word_summary, word_by_category, category_distribution, word_aggregate = (
        build_word_count_outputs(df, args)
    )
    write_csv(word_summary, args.output_dir / "word_count_summary.csv")
    write_csv(word_by_category, args.output_dir / "word_count_by_category.csv")
    write_csv(category_distribution, args.output_dir / "category_distribution.csv")

    print("Computing description-content similarity ...")
    similarity_summary, similarity_by_category, similarity_sample = compute_similarity(
        df, args
    )
    write_json(similarity_summary, args.output_dir / "text_similarity_summary.json")
    write_csv(similarity_by_category, args.output_dir / "text_similarity_by_category.csv")
    write_csv(
        similarity_sample[
            [
                args.label_column,
                "tfidf_cosine_same_row",
                "tfidf_cosine_shuffled_content",
                "token_jaccard",
                "description_token_coverage_in_content",
                "content_token_coverage_in_description",
            ]
        ],
        args.output_dir / "text_similarity_sample.csv",
    )

    classifier_metrics = None
    classification_reports = None
    classifier_comparison = None
    if not args.skip_model:
        print("Training/evaluating TF-IDF + LinearSVC classifiers ...")
        classifier_metrics, classification_reports, classifier_comparison = (
            compare_classifiers(df, args)
        )
        write_csv(
            classifier_metrics,
            args.output_dir / "classifier_feature_comparison.csv",
        )
        write_json(
            classification_reports,
            args.output_dir / "classification_report_by_feature.json",
        )

    model_comparison = None
    if classifier_metrics is not None:
        model_comparison = {
            "classifier_metrics": classifier_metrics.to_dict(orient="records"),
            **classifier_comparison,
        }

    conclusion = make_conclusion(
        data_profile=data_profile,
        word_aggregate=word_aggregate,
        similarity_summary=similarity_summary,
        model_comparison=model_comparison,
    )

    plot_paths = try_write_plots(
        output_dir=args.output_dir,
        df=df,
        similarity_sample=similarity_sample,
        classifier_metrics=classifier_metrics,
        label_column=args.label_column,
    )

    eda_summary = {
        "data_profile": data_profile,
        "word_count_aggregate": word_aggregate,
        "similarity_summary": similarity_summary,
        "model_comparison": model_comparison,
        "conclusion": conclusion,
        "plots": plot_paths,
    }
    write_json(eda_summary, args.output_dir / "eda_summary.json")

    report_path = write_report(
        output_dir=args.output_dir,
        data_profile=data_profile,
        word_aggregate=word_aggregate,
        similarity_summary=similarity_summary,
        model_comparison=model_comparison,
        conclusion=conclusion,
    )

    print(f"Done. Report: {report_path}")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
