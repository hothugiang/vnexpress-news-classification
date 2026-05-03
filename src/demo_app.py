from pathlib import Path

import pandas as pd
import streamlit as st

from evaluate_final_classifier_argmax import (
    PROJECT_ROOT,
    load_bert_model_dirs,
    load_tfidf_or_xgb_models,
    predict_final_labels,
    score_bert,
    score_tfidf_or_xgb,
)


MODEL_DIRS = {
    "tfidf_lr": PROJECT_ROOT / "models" / "one_vs_other_tfidf_lr",
    "tfidf_xgboost": PROJECT_ROOT / "models" / "one_vs_other_tfidf_xgboost",
    "bert": PROJECT_ROOT / "models" / "bert_one_vs_other",
}

STACKING_MODEL_DIRS = {
    "tfidf_lr": PROJECT_ROOT / "data" / "final_reports" / "tfidf_lr_stacking",
    "tfidf_xgboost": PROJECT_ROOT / "data" / "final_reports" / "tfidf_xgboost_stacking",
    "bert": PROJECT_ROOT / "data" / "final_reports" / "bert_stacking",
}

MODEL_LABELS = {
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "tfidf_xgboost": "TF-IDF + XGBoost",
    "bert": "BERT / PhoBERT",
}


def build_demo_text(title: str, description: str) -> str:
    return f"{(title or '').strip()} {(description or '').strip()}".strip()


@st.cache_resource(show_spinner=False)
def load_stage1_family(model_type: str):
    model_dir = MODEL_DIRS[model_type]
    if model_type in {"tfidf_lr", "tfidf_xgboost"}:
        models = load_tfidf_or_xgb_models(model_dir)
        categories = [item["target_category"] for item in models]
        return {"type": model_type, "models": models, "categories": categories}

    model_items = load_bert_model_dirs(model_dir)
    categories = [item["target_category"] for item in model_items]
    return {"type": model_type, "model_dir": model_dir, "categories": categories}


@st.cache_resource(show_spinner=False)
def load_stacking_model(model_type: str):
    import joblib

    model_dir = STACKING_MODEL_DIRS[model_type]
    model_path = model_dir / "stage2_logistic_regression.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy stage 2 model: {model_path}. "
            f"Hãy chạy evaluate_final_classifier_stacking.py trước."
        )
    return joblib.load(model_path)


def score_single_input(model_type: str, text: str, bert_batch_size: int):
    family = load_stage1_family(model_type)
    texts = [text]

    if model_type in {"tfidf_lr", "tfidf_xgboost"}:
        score_by_category = score_tfidf_or_xgb(family["models"], texts)
    else:
        score_by_category = score_bert(MODEL_DIRS[model_type], texts, bert_batch_size)

    categories = family["categories"]
    stage1_scores = {category: float(score_by_category[category][0]) for category in categories}
    return categories, stage1_scores


def run_max_voting(categories, stage1_scores, threshold: float, other_label: str):
    score_by_category = {category: [stage1_scores[category]] for category in categories}
    prediction = predict_final_labels(score_by_category, categories, threshold, other_label)[0]
    final_label, best_score, _, raw_best_category = prediction
    result_probs = {category: stage1_scores[category] for category in categories}
    return {
        "final_label": final_label,
        "final_score": best_score,
        "raw_best_category": raw_best_category,
        "threshold_applied": final_label != raw_best_category,
        "probabilities": result_probs,
    }


def run_stacking(model_type: str, categories, stage1_scores, threshold: float, other_label: str):
    stage2_model = load_stacking_model(model_type)
    feature_row = [[stage1_scores[category] for category in categories]]
    proba_row = stage2_model.predict_proba(feature_row)[0]
    stage2_classes = list(stage2_model.classes_)
    class_prob_map = {
        category: float(prob) for category, prob in zip(stage2_classes, proba_row)
    }
    raw_best_category = max(stage2_classes, key=lambda category: class_prob_map[category])
    raw_best_score = class_prob_map[raw_best_category]
    final_label = raw_best_category if raw_best_score >= threshold else other_label
    return {
        "final_label": final_label,
        "final_score": raw_best_score,
        "raw_best_category": raw_best_category,
        "threshold_applied": final_label != raw_best_category,
        "probabilities": class_prob_map,
    }


def render_probability_table(probabilities: dict, score_column_name: str):
    rows = [
        {"category": category, score_column_name: float(score)}
        for category, score in probabilities.items()
    ]
    df = pd.DataFrame(rows).sort_values(score_column_name, ascending=False).reset_index(drop=True)
    return df


def main():
    st.set_page_config(page_title="VNExpress Demo", page_icon="📰", layout="wide")
    st.title("Demo phân loại category VNExpress")
    st.caption(
        "Nhập tiêu đề và mô tả, chọn họ model stage 1, chọn cơ chế stage 2, "
        "chỉnh threshold và xem xác suất theo từng category."
    )

    with st.sidebar:
        st.header("Cấu hình")
        model_type = st.selectbox(
            "Model nhị phân stage 1",
            options=["tfidf_lr", "tfidf_xgboost", "bert"],
            format_func=lambda x: MODEL_LABELS[x],
        )
        decision_mode = st.radio(
            "Cơ chế stage 2",
            options=["max_voting", "stacking"],
            format_func=lambda x: "Max voting" if x == "max_voting" else "Stacking",
        )
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        other_label = st.text_input("Nhãn fallback", value="Khác")
        bert_batch_size = st.number_input(
            "BERT batch size",
            min_value=1,
            max_value=256,
            value=32,
            step=1,
            help="Chỉ ảnh hưởng khi chọn model BERT.",
        )

        if decision_mode == "stacking":
            stacking_model_path = STACKING_MODEL_DIRS[model_type] / "stage2_logistic_regression.joblib"
            if stacking_model_path.exists():
                st.success(f"Tìm thấy stage 2 model: {stacking_model_path.name}")
            else:
                st.error(
                    "Chưa có stage 2 model cho lựa chọn này. "
                    "Hãy chạy evaluate_final_classifier_stacking.py trước."
                )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        title = st.text_input("Tiêu đề", placeholder="Nhập tiêu đề bài báo")
        description = st.text_area("Mô tả", placeholder="Nhập mô tả bài báo", height=220)
        run_clicked = st.button("Chạy dự đoán", type="primary", use_container_width=True)

    with col_right:
        st.markdown("**Thiết lập hiện tại**")
        st.write(
            {
                "model_type": model_type,
                "stage_2_method": decision_mode,
                "threshold": threshold,
                "other_label": other_label,
            }
        )

    if not run_clicked:
        st.info("Nhập tiêu đề và mô tả, sau đó bấm `Chạy dự đoán`.")
        return

    text = build_demo_text(title, description)
    if not text:
        st.error("Cần nhập ít nhất tiêu đề hoặc mô tả.")
        return

    with st.spinner("Đang chạy stage 1 ..."):
        categories, stage1_scores = score_single_input(model_type, text, int(bert_batch_size))

    if decision_mode == "max_voting":
        final_result = run_max_voting(categories, stage1_scores, threshold, other_label)
        final_table = render_probability_table(final_result["probabilities"], "stage1_score")
    else:
        try:
            with st.spinner("Đang chạy stage 2 stacking ..."):
                final_result = run_stacking(
                    model_type=model_type,
                    categories=categories,
                    stage1_scores=stage1_scores,
                    threshold=threshold,
                    other_label=other_label,
                )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        final_table = render_probability_table(final_result["probabilities"], "stage2_probability")

    st.subheader("Kết quả cuối cùng")
    st.metric("Category dự đoán", final_result["final_label"])
    st.write(
        {
            "raw_best_category": final_result["raw_best_category"],
            "best_score": round(float(final_result["final_score"]), 6),
            "threshold_applied": bool(final_result["threshold_applied"]),
        }
    )

    st.subheader("Xác suất / score theo từng category")
    st.dataframe(final_table, use_container_width=True)

    with st.expander("Chi tiết output stage 1", expanded=(decision_mode == "stacking")):
        stage1_table = render_probability_table(stage1_scores, "stage1_score")
        st.dataframe(stage1_table, use_container_width=True)


if __name__ == "__main__":
    main()
