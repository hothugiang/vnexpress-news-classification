#!/usr/bin/env python3
"""Parse best_metrics.json from compare_pretrain.sh runs and print mean±std table.

Usage:
    python scripts/report_pretrain.py --output_base output/compare
    python scripts/report_pretrain.py --output_base output/compare --metrics recall@1 recall@10 ndcg@10
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


METRICS = [
    "test/recall@1",
    "test/recall@10",
    "test/recall@50",
    "test/ndcg@1",
    "test/ndcg@10",
    "test/ndcg@50",
    "test/mrr@1",
    "test/mrr@10",
    "test/mrr@50",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base", type=str, default="output/compare")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Subset of metrics to show (e.g. recall@1 recall@10 ndcg@10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = [f"test/{m}" if not m.startswith("test/") else m
               for m in args.metrics] if args.metrics else METRICS

    if not os.path.isdir(args.output_base):
        print(f"[ERROR] Directory not found: {args.output_base}")
        return

    results = defaultdict(list)

    for dirname in sorted(os.listdir(args.output_base)):
        path = os.path.join(args.output_base, dirname, "best_metrics.json")
        if not os.path.isfile(path):
            print(f"  [MISSING] {dirname}/best_metrics.json")
            continue
        with open(path) as f:
            data = json.load(f)
        # strip trailing -seed{N} to group runs: works for both
        # mscrs-pretrain-seed1  and  dcmome-pretrain-LR7e-4-seed2
        model = re.sub(r"-seed\d+$", "", dirname)
        results[model].append(data)

    if not results:
        print("No best_metrics.json files found.")
        return

    col_w = 14
    short = [m.replace("test/", "") for m in metrics]
    header = f"{'Model':<12}  {'Seeds':>7}  {'BestEp':>6}" + "".join(f"  {s:>{col_w}}" for s in short)
    print(header)
    print("-" * len(header))

    for model, runs in sorted(results.items()):
        seeds = [str(r.get("seed", "?")) for r in runs]
        epochs = [r.get("epoch", "?") for r in runs]
        seed_str = ",".join(seeds)
        epoch_str = ",".join(str(e) for e in epochs)

        row = f"{model:<12}  {seed_str:>7}  {epoch_str:>6}"
        for m in metrics:
            vals = [r[m] for r in runs if m in r]
            if vals:
                mean = np.mean(vals) * 100
                std = np.std(vals) * 100
                cell = f"{mean:.2f}±{std:.2f}"
                row += f"  {cell:>{col_w}}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    print()
    print(f"({len(results)} models, metrics multiplied by 100)")


if __name__ == "__main__":
    main()
