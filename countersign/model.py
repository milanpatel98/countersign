"""The model: logistic regression, trained with numpy, explained per decision.

Why not a bigger model? Card issuers have to tell a customer why a payment was
declined, and a linear model gives exact reasons: each feature's contribution
to the score is just weight times value. The evaluation compares it against
hand-written rules so the gain is measured, not assumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .features import LABELS, NAMES


@dataclass
class Model:
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    review_at: float = 0.5  # send to a human above this
    decline_at: float = 0.9  # block outright above this
    meta: dict = field(default_factory=dict)

    def logit(self, x: list[float]) -> float:
        s = self.bias
        for i, v in enumerate(x):
            s += self.weights[i] * (v - self.mean[i]) / self.std[i]
        return s

    def score(self, x: list[float]) -> float:
        z = self.logit(x)
        return 1.0 / (1.0 + math.exp(-z)) if z > -60 else 0.0

    def decide(self, p: float) -> str:
        if p >= self.decline_at:
            return "decline"
        if p >= self.review_at:
            return "review"
        return "approve"

    def reasons(self, x: list[float], k: int = 3) -> list[dict]:
        """Top contributions pushing this score up, in plain words."""
        contrib = [(self.weights[i] * (v - self.mean[i]) / self.std[i], i) for i, v in enumerate(x)]
        contrib.sort(reverse=True)
        return [{"feature": NAMES[i], "label": LABELS[NAMES[i]], "weight": round(c, 3)} for c, i in contrib[:k] if c > 0.25]

    def to_json(self) -> str:
        return json.dumps(
            {
                "features": NAMES,
                "mean": self.mean,
                "std": self.std,
                "weights": self.weights,
                "bias": self.bias,
                "review_at": self.review_at,
                "decline_at": self.decline_at,
                "meta": self.meta,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Model":
        d = json.loads(text)
        return cls(d["mean"], d["std"], d["weights"], d["bias"], d["review_at"], d["decline_at"], d.get("meta", {}))


def train(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, epochs: int = 400, lr: float = 0.5, fraud_weight: float = 20.0) -> Model:
    """Full-batch gradient descent with class weighting. Deterministic: no shuffling."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    w = np.zeros(X.shape[1])
    b = 0.0
    sw = np.where(y == 1, fraud_weight, 1.0)
    sw = sw / sw.mean()
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    # Undo the class weighting in the intercept so scores read as probabilities.
    b -= math.log(fraud_weight)
    return Model([float(v) for v in mean], [float(v) for v in std], [float(v) for v in w], float(b))


def scores(model: Model, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(model.mean)) / np.array(model.std)
    return 1.0 / (1.0 + np.exp(-(Z @ np.array(model.weights) + model.bias)))
