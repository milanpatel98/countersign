"""Evaluation the way a fraud team reads it.

* Split by time, never at random: train on days 0-19, pick thresholds on days
  20-23, report on days 24-29. A random split lets the model peek at the future.
* The operating point is an alert budget: analysts can review about 0.5% of
  traffic. What matters is how much fraud, and how many dollars of it, land
  inside that budget.
* Baselines: hand-written rules a team would start with, and the same model
  trained on features rebuilt in event-time order (the skewed way).
"""

from __future__ import annotations

import numpy as np

from .features import NAMES, replay
from .gen import DAY, generate
from .model import Model, scores, train

ALERT_BUDGET = 0.005


def rules_score(X: np.ndarray) -> np.ndarray:
    """What a fraud team writes on day one. Higher is more suspicious."""
    f = {n: X[:, i] for i, n in enumerate(NAMES)}
    s = np.zeros(len(X))
    s += (f["tiny_txns_last_10m"] >= 3) * 3
    s += (f["new_device"] == 1) * (np.expm1(f["log_amount"]) > 200) * 2
    s += (f["travel_speed"] > np.log1p(800)) * 3
    s += (f["gift_card"] == 1) * (np.expm1(f["log_amount"]) > 300) * 1
    s += (f["txns_last_10m"] >= 4) * 1
    return s + 1e-6 * f["log_amount"]  # break ties by amount


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(1, y.sum()))


def at_budget(y: np.ndarray, s: np.ndarray, amount: np.ndarray, budget: float = ALERT_BUDGET) -> dict:
    k = max(1, int(round(len(y) * budget)))
    top = np.argsort(-s, kind="stable")[:k]
    caught = y[top].sum()
    return {
        "reviewed": int(k),
        "fraud_caught": int(caught),
        "fraud_total": int(y.sum()),
        "recall": float(caught / max(1, y.sum())),
        "precision": float(caught / k),
        "dollars_caught": float(amount[top][y[top] == 1].sum()),
        "dollars_total": float(amount[y == 1].sum()),
        "false_alarms_per_10k": float((k - caught) / len(y) * 10_000),
    }


def threshold_for_budget(s: np.ndarray, budget: float) -> float:
    return float(np.quantile(s, 1 - budget))


def matrix(rows):
    X = np.array([f for _, f in rows], dtype=float)
    y = np.array([1 if e.fraud else 0 for e, _ in rows])
    amount = np.array([e.amount for e, _ in rows])
    day = np.array([int(e.event_time // DAY) for e, _ in rows])
    pattern = np.array([e.pattern for e, _ in rows])
    return X, y, amount, day, pattern


def run(seed: int = 7) -> tuple[Model, dict]:
    data = generate(seed)
    served = replay(data.events, data.homes, "arrival")
    tidy = replay(data.events, data.homes, "event")
    X, y, amount, day, pattern = matrix(served)

    # The same transactions with tidily rebuilt features, aligned by id.
    by_id = {e.txn_id: f for e, f in tidy}
    X_tidy = np.array([by_id[e.txn_id] for e, _ in served], dtype=float)
    differs = int((X != X_tidy).any(axis=1).sum())

    tr, va, te = day < 20, (day >= 20) & (day < 24), day >= 24
    model = train(X[tr], y[tr])
    model_tidy = train(X_tidy[tr], y[tr])

    s_va = scores(model, X[va])
    model.review_at = threshold_for_budget(s_va, ALERT_BUDGET)
    # Decline only where validation precision is at least 90%.
    order = np.argsort(-s_va)
    prec = np.cumsum(y[va][order]) / np.arange(1, va.sum() + 1)
    ok = np.where(prec >= 0.9)[0]
    model.decline_at = float(max(s_va[order][ok[-1]], model.review_at)) if len(ok) else 0.999

    s_te = scores(model, X[te])
    s_te_skew = scores(model_tidy, X[te])  # trained on tidy features, served live ones
    s_rules = rules_score(X[te])

    per_pattern = {}
    top = np.argsort(-s_te)[: max(1, int(round(te.sum() * ALERT_BUDGET)))]
    flagged = np.zeros(te.sum(), dtype=bool)
    flagged[top] = True
    for p in ["testing", "takeover", "skimmed", "quiet"]:
        m = pattern[te] == p
        per_pattern[p] = {"total": int(m.sum()), "caught": int((flagged & m).sum())}

    decisions = np.array([model.decide(p) for p in s_te])
    report = {
        "transactions": int(len(y)),
        "fraud": int(y.sum()),
        "late_events": int(sum(1 for e in data.events if e.arrival_time - e.event_time > 30)),
        "duplicates": int(len(data.events) - len(served)),
        "skewed_vectors": differs,
        "split": {"train": int(tr.sum()), "validate": int(va.sum()), "test": int(te.sum()), "test_fraud": int(y[te].sum())},
        "model": {"pr_auc": pr_auc(y[te], s_te), **at_budget(y[te], s_te, amount[te])},
        "rules": {"pr_auc": pr_auc(y[te], s_rules), **at_budget(y[te], s_rules, amount[te])},
        "skewed_training": {"pr_auc": pr_auc(y[te], s_te_skew), **at_budget(y[te], s_te_skew, amount[te])},
        "by_pattern": per_pattern,
        "decisions": {
            k: {"count": int((decisions == k).sum()), "fraud": int(y[te][decisions == k].sum())}
            for k in ("approve", "review", "decline")
        },
        "weights": dict(zip(NAMES, [round(w, 3) for w in model.weights])),
    }
    model.meta = {"seed": seed, "trained_on_days": "0-19", "alert_budget": ALERT_BUDGET}
    return model, report
