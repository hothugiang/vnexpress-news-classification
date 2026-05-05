# VNExpress Category Classification

Project này xây dựng bài toán phân loại 14 category bài báo VNExpress theo hướng:

1. `Stage 1`: tạo `14` bộ phân loại nhị phân `one-vs-other`
2. `Stage 2`: tạo bộ phân loại cuối cùng từ output của `14` bộ phân loại stage 1

Hiện tại repo có 3 họ model cho `stage 1`:

- `TF-IDF + Logistic Regression`
- `TF-IDF + XGBoost`
- `BERT / PhoBERT`

Và 2 cơ chế cho `stage 2`:

- `max voting`
- `stacking (Logistic Regression)`

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
│   ├── evaluate_one_vs_other_tfidf_lr.py
│   ├── train_one_vs_other_tfidf_xgboost.py
│   ├── evaluate_one_vs_other_tfidf_xgboost.py
│   ├── train_bert_one_vs_other.py
│   ├── evaluate_bert_one_vs_other.py
│   ├── evaluate_final_classifier_argmax.py
│   ├── evaluate_final_classifier_stacking.py
│   └── compare_models.py
├── data/
│   ├── split/
│   ├── one_vs_other/
│   ├── test/
│   ├── reports_tfidf_lr/
│   ├── reports_tfidf_xgboost/
│   ├── reports_bert/
│   └── final_reports/
└── models/
    ├── one_vs_other_tfidf_lr/
    ├── one_vs_other_tfidf_xgboost/
    └── bert_one_vs_other/
```

`data/` và `models/` là các thư mục output được sinh ra khi chạy script.

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

Cài thư viện cơ bản:

```powershell
python -m pip install --upgrade pip
python -m pip install scikit-learn joblib xgboost-gpu
```

Nếu dùng BERT:

```powershell
python -m pip install torch transformers sentencepiece
```

Nếu không muốn activate môi trường:

```powershell
.\.venv\Scripts\python.exe <script.py>
```

## Dữ liệu đầu vào

- File dữ liệu đã clean tại root project:

```text
data_cleaned.csv
```

- Thống kê mô tả dataset:

```text
VNExpress_Dataset.pdf
```

File `data_cleaned.csv` cần có ít nhất các cột:

- `title`
- `title_clean`
- `description`
- `main_category`

## Chuẩn bị dữ liệu

### 1. Chia train/test 80/20 theo từng category

Script:

```text
src/prepare_data/prepare_one_vs_other_data.py
```

Command:

```powershell
python src\prepare_data\prepare_one_vs_other_data.py
```

Output:

```text
data/split/train.csv
data/split/test.csv
data/split/summary.json
```

### 2. Tạo các bộ one-vs-other cho train và test

Script:

```text
src/prepare_data/create_one_vs_other_data.py
```

Command:

```powershell
python src\prepare_data\create_one_vs_other_data.py
```

Output:

```text
data/one_vs_other/*.csv
data/one_vs_other/summary.json
data/test/*.csv
data/test/summary.json
```

Trong đó:

- `data/one_vs_other/*.csv`: dùng để train 14 binary classifiers
- `data/test/*.csv`: dùng để evaluate từng binary classifier

Mỗi file trong `data/test` là một bộ test nhị phân có:

- `target_category`
- `binary_label`

## Stage 1 - TF-IDF + Logistic Regression

### Train

Script:

```text
src/train_one_vs_other_tfidf_lr.py
```

Command mẫu:

```powershell
python src\train_one_vs_other_tfidf_lr.py --input-dir data\one_vs_other --output-dir models\one_vs_other_tfidf_lr --max-features 100000 --max-iter 1000 --seed 42
```

Output:

```text
models/one_vs_other_tfidf_lr/*.joblib
models/one_vs_other_tfidf_lr/summary.json
```

### Evaluate từng binary classifier

Script:

```text
src/evaluate_one_vs_other_tfidf_lr.py
```

Command:

```powershell
python src\evaluate_one_vs_other_tfidf_lr.py --model-dir models\one_vs_other_tfidf_lr --test-dir data\test --output-dir data\reports_tfidf_lr --label-column main_category
```

Output:

```text
data/reports_tfidf_lr/individual_binary_metrics.csv
data/reports_tfidf_lr/individual_binary_metrics.json
data/reports_tfidf_lr/summary.json
```

## Stage 1 - TF-IDF + XGBoost

### Train

Script:

```text
src/train_one_vs_other_tfidf_xgboost.py
```

Command mẫu:

```powershell
python src\train_one_vs_other_tfidf_xgboost.py --input-dir data\one_vs_other --output-dir models\one_vs_other_tfidf_xgboost --max-features 100000 --n-estimators 200 --max-depth 4 --learning-rate 0.05 --subsample 0.8 --colsample-bytree 0.8 --reg-lambda 1.0 --tree-method hist --n-jobs -1 --seed 42
```

Output:

```text
models/one_vs_other_tfidf_xgboost/*.joblib
models/one_vs_other_tfidf_xgboost/summary.json
```

### Evaluate từng binary classifier

Script:

```text
src/evaluate_one_vs_other_tfidf_xgboost.py
```

Command:

```powershell
python src\evaluate_one_vs_other_tfidf_xgboost.py --model-dir models\one_vs_other_tfidf_xgboost --test-dir data\test --output-dir data\reports_tfidf_xgboost --label-column main_category
```

Output:

```text
data/reports_tfidf_xgboost/individual_binary_metrics.csv
data/reports_tfidf_xgboost/individual_binary_metrics.json
data/reports_tfidf_xgboost/summary.json
```

## Stage 1 - BERT / PhoBERT

### Train

Script:

```text
src/train_bert_one_vs_other.py
```

Command mẫu:

```powershell
python src\train_bert_one_vs_other.py --input-dir data\one_vs_other --output-dir models\bert_one_vs_other --model-name vinai/phobert-base-v2 --epochs 3 --batch-size 32 --lr 2e-5 --max-len 128
```

Train một category để thử nghiệm nhanh:

```powershell
python src\train_bert_one_vs_other.py --category thu_gian
```

Output:

```text
models/bert_one_vs_other/
    <category>/
        config.json
        model.safetensors
        tokenizer files
        meta.json
    summary.json
```

### Evaluate từng binary classifier

Script:

```text
src/evaluate_bert_one_vs_other.py
```

Command:

```powershell
python src\evaluate_bert_one_vs_other.py --model-dir models\bert_one_vs_other --test-dir data\test --output-dir data\reports_bert
```

Output:

```text
data/reports_bert/individual_binary_metrics.csv
data/reports_bert/individual_binary_metrics.json
data/reports_bert/summary.json
```

## So sánh 3 họ model ở stage 1

Script:

```text
src/compare_models.py
```

Command:

```powershell
python src\compare_models.py
```

Output:

```text
data/model_comparison/per_category_comparison.csv
data/model_comparison/macro_summary.csv
data/model_comparison/summary.json
```

## Stage 2 - Final classifier

Stage 2 nhận input là output của `14` bộ phân loại nhị phân từ stage 1.

Hiện tại có 2 cơ chế:

1. `max voting`
2. `stacking`

Cả hai đều hỗ trợ `threshold`:

- nếu không có class nào vượt threshold
- dự đoán cuối cùng sẽ được đưa về nhãn `Khác`

### Stage 2 - Max voting

Script:

```text
src/evaluate_final_classifier_argmax.py
```

Logic:

- chạy mỗi bài test qua 14 model stage 1
- lấy 14 score / probability
- chọn nhãn có score cao nhất
- nếu `max_score < threshold` thì đưa về `Khác`

#### Command cho TF-IDF + Logistic Regression

```powershell
python src\evaluate_final_classifier_argmax.py --model-type tfidf_lr --threshold 0.5 --other-label "Khác"
```

#### Command cho TF-IDF + XGBoost

```powershell
python src\evaluate_final_classifier_argmax.py --model-type tfidf_xgboost --threshold 0.5 --other-label "Khác"
```

#### Command cho BERT

```powershell
python src\evaluate_final_classifier_argmax.py --model-type bert --threshold 0.5 --other-label "Khác" --bert-batch-size 64
```

Output mặc định:

```text
data/final_reports/tfidf_lr_maxvoting/
data/final_reports/tfidf_xgboost_maxvoting/
data/final_reports/bert_maxvoting/
```

### Stage 2 - Stacking

Script:

```text
src/evaluate_final_classifier_stacking.py
```

Logic:

- dùng `data/split/train.csv` và `data/split/test.csv`
- sinh feature 14 chiều từ output của 14 model stage 1
- train `Logistic Regression` đa lớp trên train split
- predict trên test split
- nếu `max_proba_stage2 < threshold` thì đưa về `Khác`

Lưu ý:

- bản stacking hiện tại không dùng OOF
- đây là một biến thể thực nghiệm của stage 2

#### Command cho TF-IDF + Logistic Regression

```powershell
python src\evaluate_final_classifier_stacking.py --model-type tfidf_lr --threshold 0.5 --other-label "Khác" --max-iter 1000 --seed 42
```

#### Command cho TF-IDF + XGBoost

```powershell
python src\evaluate_final_classifier_stacking.py --model-type tfidf_xgboost --threshold 0.5 --other-label "Khác" --max-iter 1000 --seed 42
```

#### Command cho BERT

```powershell
python src\evaluate_final_classifier_stacking.py --model-type bert --threshold 0.5 --other-label "Khác" --bert-batch-size 64 --max-iter 1000 --seed 42
```

Output mặc định:

```text
data/final_reports/tfidf_lr_stacking/
data/final_reports/tfidf_xgboost_stacking/
data/final_reports/bert_stacking/
```

## Demo giao diện

Project có một giao diện demo đơn giản để chạy thử dự đoán trên một input mới.

Script:

```text
src/demo_app.py
```

Giao diện cho phép:

- nhập `tiêu đề`
- nhập `mô tả`
- chọn model nhị phân ở stage 1:
  - `TF-IDF + Logistic Regression`
  - `TF-IDF + XGBoost`
  - `BERT / PhoBERT`
- chọn cơ chế stage 2:
  - `max voting`
  - `stacking`
- chọn `threshold`
- chọn nhãn fallback, mặc định là `Khác`

Output trả về gồm:

- category dự đoán cuối cùng
- `raw_best_category`
- `best_score`
- `threshold_applied`
- bảng xác suất / score theo từng category

### Cài thêm thư viện cho demo

```powershell
python -m pip install streamlit pandas
```

### Chạy demo bằng `.venv`

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\demo_app.py
```

### Chạy demo bằng `.venv_gpu`

Nếu bạn có môi trường GPU riêng cho BERT:

```powershell
.\.venv_gpu\Scripts\python.exe -m streamlit run src\demo_app.py
```

### Lưu ý khi dùng demo

- Nếu chọn `max voting`, app sẽ dùng trực tiếp output của 14 model stage 1 để ra dự đoán cuối.
- Nếu chọn `stacking`, app cần file model stage 2 đã được tạo sẵn:

```text
data/final_reports/tfidf_lr_stacking/stage2_logistic_regression.joblib
data/final_reports/tfidf_xgboost_stacking/stage2_logistic_regression.joblib
data/final_reports/bert_stacking/stage2_logistic_regression.joblib
```

- Nếu các file này chưa tồn tại, hãy chạy `evaluate_final_classifier_stacking.py` trước.
- Nếu chọn `bert`, tốc độ sẽ chậm hơn rõ rệt so với `tfidf_lr` và `tfidf_xgboost`.

## Metric ở stage 2

Các file `metrics.json` trong `data/final_reports/*` lưu:

- `accuracy`
- `micro_precision`
- `micro_recall`
- `micro_f1`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_precision`
- `weighted_recall`
- `weighted_f1`
- `threshold_fallback_count`

Muốn đọc chi tiết hơn, xem:

- `classification_report.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `predictions.csv`

Tài liệu giải thích chi tiết các file này:

```text
data/final_reports/explain_final_reports.md
```

## Ghi chú

- `data/split/test.csv` là test set gốc sau khi chia 80/20.
- `data/one_vs_other/*.csv` dùng để train 14 binary classifiers.
- `data/test/*.csv` là test one-vs-other cho từng category, dùng để evaluate từng model stage 1.
- `accuracy` có thể cao dù model chưa tốt nếu dữ liệu mất cân bằng, nên xem thêm `precision`, `recall`, `f1`.
- với stage 2, nên đối chiếu thêm `micro_f1`, `macro_f1`, `weighted_f1`.
- model BERT gợi ý: `vinai/phobert-base-v2`, `xlm-roberta-base`.
