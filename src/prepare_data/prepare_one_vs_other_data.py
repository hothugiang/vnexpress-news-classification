import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.prepare_data.create_one_vs_other_data import create_one_vs_other_datasets


def load_rows(input_file: Path, label_column: str):
    """Đọc file CSV đã clean và gom index của các dòng theo category."""
    rows = []
    by_category = defaultdict(list)

    with input_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if label_column not in fieldnames:
            raise ValueError(f"Column '{label_column}' not found in {input_file}")

        for idx, row in enumerate(reader):
            category = (row.get(label_column) or "").strip()
            row[label_column] = category
            rows.append(row)
            by_category[category].append(idx)

    return rows, by_category, fieldnames


def write_rows(output_path: Path, rows, row_indices, fieldnames):
    """Ghi các dòng được chọn ra CSV và giữ nguyên các cột gốc."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in row_indices:
            writer.writerow(rows[idx])


def stratified_train_test_split(by_category, test_size: float):
    """Chia index thành train/test nhưng vẫn giữ tỷ lệ của từng category."""
    train_indices = []
    test_indices = []
    split_summary = []

    # Chia riêng bên trong từng category, sau đó gộp index lại.
    for category, indices in sorted(
        by_category.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        shuffled_indices = list(indices)
        random.shuffle(shuffled_indices)

        test_count = round(len(shuffled_indices) * test_size)
        test_category_indices = shuffled_indices[:test_count]
        train_category_indices = shuffled_indices[test_count:]

        test_indices.extend(test_category_indices)
        train_indices.extend(train_category_indices)

        split_summary.append(
            {
                "category": category,
                "total": len(shuffled_indices),
                "train": len(train_category_indices),
                "test": len(test_category_indices),
            }
        )

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return train_indices, test_indices, split_summary


def write_split_summary(
    output_path: Path,
    input_file: Path,
    train_path: Path,
    test_path: Path,
    label_column: str,
    test_size: float,
    seed: int,
    split_summary,
):
    """Lưu thông tin train/test split để có thể kiểm tra và tái lập."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "source": str(input_file),
                "label_column": label_column,
                "split": {
                    "train": str(train_path),
                    "test": str(test_path),
                    "train_ratio": 1 - test_size,
                    "test_ratio": test_size,
                },
                "categories": split_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def prepare_data(
    input_file: Path,
    split_dir: Path,
    one_vs_other_dir: Path,
    label_column: str,
    test_size: float,
    seed: int,
):
    """Chạy toàn bộ pipeline từ dữ liệu clean gốc tới các bộ train nhị phân."""
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    random.seed(seed)
    rows, by_category, fieldnames = load_rows(input_file, label_column)

    # Bước 1: tạo train/test split và giữ phân bố gần với dữ liệu gốc.
    train_indices, test_indices, split_summary = stratified_train_test_split(
        by_category=by_category,
        test_size=test_size,
    )

    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"

    write_rows(train_path, rows, train_indices, fieldnames)
    write_rows(test_path, rows, test_indices, fieldnames)

    # Lưu thông tin split trước khi tạo các bộ train one-vs-other.
    write_split_summary(
        output_path=split_dir / "summary.json",
        input_file=input_file,
        train_path=train_path,
        test_path=test_path,
        label_column=label_column,
        test_size=test_size,
        seed=seed,
        split_summary=split_summary,
    )

    # Bước 2: chỉ dùng train.csv để tạo các bộ one-vs-other cân bằng.
    one_vs_other_summary = create_one_vs_other_datasets(
        input_file=train_path,
        output_dir=one_vs_other_dir,
        label_column=label_column,
        seed=seed,
    )

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "split_summary": split_summary,
        "one_vs_other_summary": one_vs_other_summary,
    }


def parse_args():
    """Đọc tham số dòng lệnh cho toàn bộ pipeline chuẩn bị dữ liệu."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified train/test split, then create balanced "
            "one-vs-other training datasets from the train split."
        )
    )
    parser.add_argument(
        "--input",
        default="data_cleaned.csv",
        type=Path,
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--split-dir",
        default=Path("data") / "split",
        type=Path,
        help="Directory where train.csv and test.csv will be written.",
    )
    parser.add_argument(
        "--one-vs-other-dir",
        default=Path("data") / "one_vs_other",
        type=Path,
        help="Directory where one-vs-other train CSV files will be written.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Column used as the category label.",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Fraction of each category to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for splitting and negative sampling.",
    )
    return parser.parse_args()


def main():
    """Hàm chính khi chạy trực tiếp file này từ dòng lệnh."""
    args = parse_args()
    result = prepare_data(
        input_file=args.input,
        split_dir=args.split_dir,
        one_vs_other_dir=args.one_vs_other_dir,
        label_column=args.label_column,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(
        f"Created train/test split: train={result['train_count']}, "
        f"test={result['test_count']}"
    )
    print(f"Train file: {result['train_path']}")
    print(f"Test file: {result['test_path']}")
    print(
        f"Created {len(result['one_vs_other_summary'])} one-vs-other "
        f"training datasets in: {args.one_vs_other_dir}"
    )

    for item in result["one_vs_other_summary"]:
        print(
            f"{item['file']}: target={item['target_category']}, "
            f"pos={item['positive']}, neg={item['negative']}, total={item['total']}"
        )


if __name__ == "__main__":
    main()
