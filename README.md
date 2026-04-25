# VNExpress Category Classification

Project này xây dựng 14 bộ phân loại nhị phân `one-vs-other` cho 14 categories bài báo VNExpress, dựa trên `title` và `description`.

Baseline hiện tại:

```text
TF-IDF + Logistic Regression
```

## Cấu trúc project

```text
.
├── data_cleaned.csv
├── src/
│   ├── prepare_data/
│   │   ├── prepare_one_vs_other_data.py
│   │   ├── create_one_vs_other_data.py
│   │   └── prepare_data.qmd
│   ├── train_one_vs_other_tfidf_lr.py
│   └── evaluate_one_vs_other_tfidf_lr.py
├── data/
│   ├── split/
│   ├── one_vs_other/
│   ├── test/
│   └── reports/
└── models/
    └── one_vs_other/
```

`data/`, `models/` là thư mục output được sinh ra khi chạy script.

## Cài đặt sau khi clone

Yêu cầu:

- Python 3.10+
- Windows PowerShell hoặc terminal tương đương

Tạo môi trường ảo:

```powershell
python -m venv .venv
```

Kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install --upgrade pip
python -m pip install scikit-learn joblib
```

Nếu không muốn activate môi trường:

```powershell
.\.venv\Scripts\python.exe <script.py>
```

## Dữ liệu đầu vào

- File dữ liệu đã clean tại root project: data_cleaned.csv
- Thống kê theo từng Categories: VNExpress_Dataset.pdf

## Chuẩn bị dữ liệu

### 1. Chia train/test 80/20 theo từng category

Script: src/prepare_data/prepare_one_vs_other_data.py

Output:

```text
data/split/train.csv
data/split/test.csv
data/split/summary.json
```

### 2. Tạo lại bộ test one-vs-other để evaluate từng model

Script: src/prepare_data/create_one_vs_other_data.py

Output:

```text
data/one_vs_other/*.csv
data/one_vs_other/summary.json
data/test/*.csv
data/test/summary.json
```

Mỗi file trong `data/test` là một bộ test nhị phân có:

- `target_category`
- `binary_label`

## Train TF-IDF + Logistic Regression

Script:

```text
src/train_one_vs_other_tfidf_lr.py
```

Command mẫu:

```powershell
python src\train_one_vs_other_tfidf_lr.py --input-dir data\one_vs_other --output-dir models\one_vs_other --max-features 100000 --max-iter 1000 --seed 42
```

Output:

```text
models/one_vs_other/*.joblib
models/one_vs_other/summary.json
```

Input text của mỗi bài:

```text
title_clean/title + description
```

## Evaluate

Script:

```text
src/evaluate_one_vs_other_tfidf_lr.py
```

Command mẫu:

```powershell
python src\evaluate_one_vs_other_tfidf_lr.py --model-dir models\one_vs_other --test-dir data\test --output-dir data\reports --label-column main_category
```

Script này đánh giá từng bộ phân loại nhị phân trên file test tương ứng trong `data/test`.

Metric được lưu:

- `accuracy`
- `precision`
- `recall`
- `f1`

Output:

```text
data/reports/individual_binary_metrics.csv
data/reports/individual_binary_metrics.json
data/reports/summary.json
```

## Ghi chú

- `data/split/test.csv` là test set gốc sau khi chia 80/20.
- `data/one_vs_other/*.csv` dùng để train 14 binary classifiers.
- `data/test/*.csv` là test one-vs-other cho từng category, dùng để evaluate từng model.
- `accuracy` có thể cao dù mô hình chưa tốt nếu dữ liệu mất cân bằng, nên xem thêm `precision`, `recall`, `f1`.
