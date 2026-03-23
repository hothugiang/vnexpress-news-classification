from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import torch
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu


class RecEvaluator:
    def __init__(self, k_list: list[int] | None = None) -> None:
        self.k_list = k_list or [1, 10, 50]
        self.reset()

    def reset(self) -> None:
        self.metric = defaultdict(float)
        self.metric["count"] = 0.0

    def update(self, ranked_item_ids: torch.Tensor, labels: torch.Tensor) -> None:
        for rank, label in zip(ranked_item_ids.tolist(), labels.tolist()):
            if label < 0:
                continue
            for k in self.k_list:
                self.metric[f"recall@{k}"] += float(label in rank[:k])
                self.metric[f"mrr@{k}"] += self._mrr(rank, label, k)
                self.metric[f"ndcg@{k}"] += self._ndcg(rank, label, k)
            self.metric["count"] += 1

    def report(self) -> dict[str, float]:
        count = max(self.metric["count"], 1.0)
        return {name: value / count for name, value in self.metric.items() if name != "count"} | {"count": self.metric["count"]}

    @staticmethod
    def _mrr(rank: list[int], label: int, k: int) -> float:
        if label in rank[:k]:
            return 1.0 / (rank.index(label) + 1)
        return 0.0

    @staticmethod
    def _ndcg(rank: list[int], label: int, k: int) -> float:
        if label in rank[:k]:
            return 1.0 / math.log2(rank.index(label) + 2)
        return 0.0


class ConvEvaluator:
    slot_pattern = re.compile(r"<movie>")

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self.metric = {
            "bleu@1": 0.0,
            "bleu@2": 0.0,
            "bleu@3": 0.0,
            "bleu@4": 0.0,
            "rouge@1": 0.0,
            "rouge@2": 0.0,
            "rouge@L": 0.0,
            "dist@1": set(),
            "dist@2": set(),
            "dist@3": set(),
            "dist@4": set(),
            "item_ratio": 0.0,
        }
        self.count = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pred_texts = self._decode_predictions(preds)
        label_texts = self._decode_labels(labels)
        for pred, label in zip(pred_texts, label_texts):
            if not pred:
                continue
            self._collect_ngram(pred)
            self._compute_item_ratio(pred)
            self._compute_bleu(pred, label)
            self._compute_rouge(pred, label)
            self.count += 1

    def report(self) -> dict[str, float]:
        denom = max(self.count, 1)
        report: dict[str, float] = {}
        for name, value in self.metric.items():
            if name.startswith("dist@"):
                report[name] = len(value) / denom
            else:
                report[name] = value / denom
        report["count"] = float(self.count)
        return report

    def _decode_predictions(self, preds: torch.Tensor) -> list[str]:
        decoded = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        cleaned = [text.replace("<pad>", "").replace("<|endoftext|>", "").strip() for text in decoded]
        return [f"System: {text}".strip() for text in cleaned]

    def _decode_labels(self, labels: torch.Tensor) -> list[str]:
        decoded = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        return [text.replace("<pad>", "").replace("<|endoftext|>", "").strip() for text in decoded]

    def _collect_ngram(self, text: str) -> None:
        tokens = text.split()
        for k in range(1, 5):
            for token in ngrams(tokens, k):
                self.metric[f"dist@{k}"].add(token)

    def _compute_item_ratio(self, text: str) -> None:
        self.metric["item_ratio"] += len(self.slot_pattern.findall(text))

    def _compute_bleu(self, pred: str, label: str) -> None:
        pred_tokens = pred.split()
        label_tokens = [label.split()]
        for k in range(4):
            weights = [0.0] * 4
            weights[k] = 1.0
            self.metric[f"bleu@{k + 1}"] += sentence_bleu(label_tokens, pred_tokens, weights=weights)

    def _compute_rouge(self, pred: str, label: str) -> None:
        pred_tokens = pred.split()
        label_tokens = label.split()
        self.metric["rouge@1"] += _rouge_n_f1(pred_tokens, label_tokens, 1)
        self.metric["rouge@2"] += _rouge_n_f1(pred_tokens, label_tokens, 2)
        self.metric["rouge@L"] += _rouge_l_f1(pred_tokens, label_tokens)


def _rouge_n_f1(pred_tokens: list[str], label_tokens: list[str], n: int) -> float:
    if len(pred_tokens) < n or len(label_tokens) < n:
        return 0.0
    pred_counts = Counter(ngrams(pred_tokens, n))
    label_counts = Counter(ngrams(label_tokens, n))
    overlap = sum((pred_counts & label_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred_counts.values()), 1)
    recall = overlap / max(sum(label_counts.values()), 1)
    return 2 * precision * recall / (precision + recall)


def _rouge_l_f1(pred_tokens: list[str], label_tokens: list[str]) -> float:
    if not pred_tokens or not label_tokens:
        return 0.0
    lcs = _longest_common_subsequence_length(pred_tokens, label_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def _longest_common_subsequence_length(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for idx, token_b in enumerate(b, start=1):
            current = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = current
    return dp[-1]
