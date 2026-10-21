"""A check on real transactions: the public credit card fraud dataset.

The dataset (Worldline and the Machine Learning Group of ULB, on Kaggle as
"Credit Card Fraud Detection") has 284,807 card payments made by European
cardholders over two days in September 2013, 492 of them fraud. To protect the
cardholders, the columns were anonymised with PCA: V1 to V28, plus Time and
Amount. There are no card ids, merchants or places.

So Countersign's per-card feature engine cannot run on it. What can be tested
is everything after the features: the numpy logistic model, the time-ordered
split, the alert-budget evaluation, threshold selection, reasons and latency.
The file is not committed (150 MB, and its licence asks you to fetch it from
the source); pass its path.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from .evaluate import ALERT_BUDGET, at_budget, pr_auc, threshold_for_budget
from .model import Model, scores, train

COLUMNS = [f"V{i}" for i in range(1, 29)] + ["log_amount"]


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, X, amount, y = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] != "Time" or header[-1] != "Class":
            raise ValueError("this does not look like creditcard.csv (expected Time ... Amount, Class)")
        for row in reader:
            t.append(float(row[0]))
            amt = float(row[29])
            X.append([float(v) for v in row[1:29]] + [float(np.log1p(amt))])
            amount.append(amt)
            y.append(int(row[30].strip('"')))
    return np.array(t), np.array(X), np.array(amount), np.array(y)


def run(path: str) -> dict:
    t, X, amount, y = load(path)
    order = np.argsort(t, kind="stable")  # the file is already in time order; make sure
    t, X, amount, y = t[order], X[order], amount[order], y[order]

    # Split by time: first 60% train, next 15% pick thresholds, last 25% test.
    n = len(y)
    a, b = int(n * 0.60), int(n * 0.75)
    tr, va, te = slice(0, a), slice(a, b), slice(b, n)

    model = train(X[tr], y[tr], epochs=600)
    s_va = scores(model, X[va])
    model.review_at = threshold_for_budget(s_va, ALERT_BUDGET)
    order_va = np.argsort(-s_va)
    prec = np.cumsum(y[va][order_va]) / np.arange(1, len(order_va) + 1)
    ok = np.where(prec >= 0.9)[0]
    model.decline_at = float(max(s_va[order_va][ok[-1]], model.review_at)) if len(ok) else 0.999

    s_te = scores(model, X[te])
    # Baseline: flag the biggest payments, which is what a team with no model often does first.
    s_amount = amount[te].astype(float)

    decisions = np.where(s_te >= model.decline_at, "decline", np.where(s_te >= model.review_at, "review", "approve"))

    # Latency of the same scalar scoring path the live scorer uses.
    rows = [list(r) for r in X[te][:20000]]
    t0 = time.perf_counter_ns()
    for r in rows:
        model.decide(model.score(r))
    per_us = (time.perf_counter_ns() - t0) / len(rows) / 1000

    return {
        "dataset": "Credit Card Fraud Detection (Worldline and ULB), 2 days of European card payments, September 2013",
        "transactions": int(n),
        "fraud": int(y.sum()),
        "split": {
            "train": int(a),
            "validate": int(b - a),
            "test": int(n - b),
            "test_fraud": int(y[te].sum()),
            "test_hours": [round(float(t[b]) / 3600, 1), round(float(t[-1]) / 3600, 1)],
        },
        "model": {"pr_auc": pr_auc(y[te], s_te), **at_budget(y[te], s_te, amount[te])},
        "largest_amounts": {"pr_auc": pr_auc(y[te], s_amount), **at_budget(y[te], s_amount, amount[te])},
        "decisions": {k: {"count": int((decisions == k).sum()), "fraud": int(y[te][decisions == k].sum())} for k in ("approve", "review", "decline")},
        "score_us_per_payment": round(per_us, 2),
        "weights": dict(zip(COLUMNS, [round(w, 3) for w in model.weights])),
    }


def model_for(path: str) -> Model:
    t, X, amount, y = load(path)
    return train(X, y, epochs=600)
