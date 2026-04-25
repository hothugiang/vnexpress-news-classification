import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def slugify(text: str) -> str:
    """Chuyển tên category tiếng Việt thành tên file ASCII an toàn."""
    text = text.translate({0x0110: "D", 0x0111: "d"})
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "category"


def load_rows(input_file: Path, label_column: str):
    """Đọc file CSV và gom index của các dòng theo nhãn category."""
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


def sample_negative_ids(rows, by_category, categories, target_category, positive_count):
    """Lấy mẫu negative từ các category khác theo tỷ lệ phân bố gốc."""
    other_categories = [category for category in categories if category != target_category]
    other_total = sum(len(by_category[category]) for category in other_categories)

    negative_ids = []
    selected_negative_ids = set()
    fractional_parts = []

    # Lượt 1: lấy phần nguyên của số lượng phân bổ theo tỷ lệ.
    for category in other_categories:
        exact_count = positive_count * len(by_category[category]) / other_total
        take_count = int(exact_count)

        sampled_ids = random.sample(by_category[category], take_count)
        negative_ids.extend(sampled_ids)
        selected_negative_ids.update(sampled_ids)
        fractional_parts.append((exact_count - take_count, category))

    remaining = positive_count - len(negative_ids)

    # Lượt 2: bù các dòng còn thiếu do bước làm tròn xuống.
    for _, category in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break

        available_ids = [
            idx for idx in by_category[category] if idx not in selected_negative_ids
        ]

        if not available_ids:
            continue

        take_count = min(remaining, len(available_ids))
        sampled_ids = random.sample(available_ids, take_count)

        negative_ids.extend(sampled_ids)
        selected_negative_ids.update(sampled_ids)
        remaining -= take_count

    # Dự phòng nếu trường hợp làm tròn hiếm gặp vẫn làm thiếu negative.
    if len(negative_ids) != positive_count:
        all_other_ids = [
            idx
            for category in other_categories
            for idx in by_category[category]
            if idx not in selected_negative_ids
        ]
        need = positive_count - len(negative_ids)
        negative_ids.extend(random.sample(all_other_ids, need))

    return negative_ids


def write_dataset(
    output_path,
    rows,
    examples,
    target_category,
    fieldnames,
):
    """Ghi một bộ dữ liệu nhị phân one-vs-other ra file CSV."""
    output_fieldnames = fieldnames + ["target_category", "binary_label"]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()

        for row_idx, label in examples:
            output_row = dict(rows[row_idx])
            output_row["target_category"] = target_category
            output_row["binary_label"] = label
            writer.writerow(output_row)


def create_one_vs_other_datasets(input_file, output_dir, label_column, seed):
    """Tạo một bộ dữ liệu nhị phân cân bằng cho từng category trong file input."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, by_category, fieldnames = load_rows(input_file, label_column)
    categories = [
        category
        for category, ids in sorted(
            by_category.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
    ]

    summary = []

    for target_category in categories:
        # Positive: toàn bộ dòng thuộc target category.
        positive_ids = list(by_category[target_category])
        positive_count = len(positive_ids)

        # Negative: lấy cùng số lượng dòng từ tất cả category còn lại.
        negative_ids = sample_negative_ids(
            rows=rows,
            by_category=by_category,
            categories=categories,
            target_category=target_category,
            positive_count=positive_count,
        )

        examples = [(idx, 1) for idx in positive_ids]
        examples += [(idx, 0) for idx in negative_ids]
        random.shuffle(examples)

        output_path = output_dir / f"{slugify(target_category)}.csv"
        write_dataset(
            output_path=output_path,
            rows=rows,
            examples=examples,
            target_category=target_category,
            fieldnames=fieldnames,
        )

        # Lưu phân bố category gốc của negative để kiểm tra/debug.
        negative_source_counts = Counter(rows[idx][label_column] for idx in negative_ids)
        summary.append(
            {
                "file": output_path.name,
                "target_category": target_category,
                "positive": positive_count,
                "negative": len(negative_ids),
                "total": len(examples),
                "negative_source_counts": dict(negative_source_counts),
            }
        )

    # Summary giúp kiểm tra từng file nhị phân đã cân bằng và có thể tái lập.
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "source": str(input_file),
                "label_column": label_column,
                "strategy": (
                    "balanced one-vs-other; all positives and proportionally "
                    "sampled negatives from other categories"
                ),
                "files": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def parse_args():
    """Đọc tham số dòng lệnh khi chỉ muốn tạo dữ liệu one-vs-other."""
    parser = argparse.ArgumentParser(
        description="Create balanced one-vs-other CSV datasets for each category."
    )
    parser.add_argument(
        "--input",
        default="data_cleaned.csv",
        type=Path,
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data") / "one_vs_other",
        type=Path,
        help="Directory where one-vs-other CSV files will be written.",
    )
    parser.add_argument(
        "--label-column",
        default="main_category",
        help="Column used as the category label.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for negative sampling.",
    )
    return parser.parse_args()


def main():
    """Hàm chính khi chạy trực tiếp file này từ dòng lệnh."""
    args = parse_args()
    summary = create_one_vs_other_datasets(
        input_file=args.input,
        output_dir=args.output_dir,
        label_column=args.label_column,
        seed=args.seed,
    )

    print(f"Created {len(summary)} one-vs-other datasets in: {args.output_dir}")
    for item in summary:
        print(
            f"{item['file']}: target={item['target_category']}, "
            f"pos={item['positive']}, neg={item['negative']}, total={item['total']}"
        )


if __name__ == "__main__":
    main()
