# EDA description vs content

## Mục tiêu

File này kiểm chứng liệu `description` có đủ đại diện cho `content` để dùng làm feature chính dự đoán `category/main_category`, đồng thời thống kê và so sánh số từ giữa hai trường văn bản.

## Dữ liệu

- Input: `data_cleaned.csv`
- Số dòng ban đầu: 71,419
- Số dòng dùng cho EDA sau khi bỏ rỗng `description`, `content`, `main_category`: 71,419

## So sánh số từ

- `description`: trung bình 29.1 từ, median 29 từ.
- `content`: trung bình 579.8 từ, median 523 từ.
- Theo median, `description` chỉ dài bằng 5.5% so với `content`.
- Có 2,727 dòng content ngắn hơn description; đây là nhóm nên kiểm tra thêm nếu cần làm sạch dữ liệu.

## Tương quan nội dung description-content

- TF-IDF cosine cùng dòng: mean 0.217, median 0.218.
- Baseline tráo `content` sang dòng khác: mean 0.019, median 0.017.
- Cùng dòng cao hơn baseline tráo dòng 11.33 lần theo mean cosine.
- Trung bình 77.4% token riêng biệt của `description` cũng xuất hiện trong `content`.
- Jaccard token trung bình là 0.084; chỉ số này thấp hơn coverage vì `content` dài hơn rất nhiều.

## Kết luận

- Chưa chạy phần classifier vì bật `--skip-model`.
- Chưa chạy classifier vì dùng --skip-model.

## Output files

- `category_distribution.csv`
- `word_count_summary.csv`
- `word_count_by_category.csv`
- `text_similarity_summary.json`
- `text_similarity_by_category.csv`
- `text_similarity_sample.csv`
- `classifier_feature_comparison.csv`
- `classification_report_by_feature.json`
- `similarity_mean_same_vs_shuffled.png`
- `similarity_key_metrics_bar.png`
- `description_token_coverage_distribution.png`
- `similarity_by_category.png`
- `description_token_coverage_by_category.png`
- `eda_summary.json`
